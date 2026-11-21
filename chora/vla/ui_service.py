"""Isolated VLA subprocess management; the regular lab does not import torch."""
import json
import hashlib
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid


def read(path, default=None):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else default
    except (OSError, ValueError):
        return default


class VLAService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.process = self.directory = self.log = self.run_id = self.stop_at = None
        self.lock = threading.RLock()
        self.weight_hashes = {}

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def evaluations(self):
        trials = {}
        folders = [self.root/'outputs/vla/rollouts', self.root/'outputs/vla/ui-runs']
        files = [f for folder in folders for f in folder.glob('*/report.json')]
        for file in sorted(files, key=lambda p:p.stat().st_mtime):
            report = read(file, {})
            checkpoint = report.get('checkpoint')
            if report.get('status') != 'complete' or not isinstance(checkpoint,str) or report.get('execute_horizon',4) != 4:
                continue
            checkpoint = str((self.root/checkpoint).resolve())
            for trial in report.get('trials',[]):
                if trial.get('reason') not in {'success','collision','tick_limit','policy_error'}:
                    continue
                seed = trial.get('seed')
                if isinstance(seed,int):
                    trials.setdefault(checkpoint,{})[seed] = {**trial,'weight_sha256':report.get('checkpoint_sha256')}
        return trials

    def candidates(self):
        training = (self.root/'outputs/vla/training').resolve()
        datasets = (self.root/'outputs/vla/datasets').resolve()
        by_fingerprint = {}
        for file in sorted(datasets.glob('*/dataset.json')):
            value = read(file, {})
            key = value.get('fingerprint')
            if value.get('protocol') in {'chora-vla-pilot-chunks-v1','chora-vla-pilot-chunks-v2'} and isinstance(key, str) and file.resolve().is_relative_to(datasets):
                by_fingerprint[key] = file.parent.resolve()
        result = []
        evaluations = self.evaluations()
        for file in training.glob('*/report.json'):
            report = read(file, {})
            if report.get('status') != 'complete' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', file.parent.name):
                continue
            relative = (report.get('final') or {}).get('checkpoint')
            if not isinstance(relative, str):
                continue
            checkpoint = (self.root/relative).resolve()
            if not checkpoint.is_relative_to(training):
                continue
            config = read(checkpoint/'config.json', {})
            metadata = config.get('metadata') or {}
            dataset = by_fingerprint.get(metadata.get('dataset_fingerprint'))
            weights = checkpoint/'model.safetensors'
            if not dataset or not weights.is_file() or not weights.resolve().is_relative_to(training):
                continue
            if config.get('protocol') != 'chora-vla-skeleton-v1' or 'action_center' not in (config.get('normalization') or {}):
                continue
            step = metadata.get('step', 0)
            if not isinstance(step, int):
                continue
            name = file.parent.name.removeprefix('bowl-').removesuffix('-v1').replace('-', ' ').capitalize()
            trials = list(evaluations.get(str(checkpoint),{}).values())
            passed = sum(t.get('success') is True for t in trials)
            evaluation = dict(successes=passed,episodes=len(trials),collisions=sum(bool(t.get('collisions')) for t in trials))
            known_seeds={e['seed'] for e in read(dataset/'dataset.json',{}).get('episodes',[])}
            unseen=[t for t in trials if t.get('seed') not in known_seeds]
            unseen_passed=sum(t.get('success') is True for t in unseen)
            evaluation.update(unseen_episodes=len(unseen),unseen_successes=unseen_passed)
            recommended = False
            if len(unseen)>=10 and unseen_passed/len(unseen)>=.8 and not any(t.get('collisions') for t in unseen):
                stat=weights.stat();key=(str(weights),stat.st_size,stat.st_mtime_ns)
                if key not in self.weight_hashes:self.weight_hashes[key]=hashlib.sha256(weights.read_bytes()).hexdigest()
                recommended=all(t.get('weight_sha256')==self.weight_hashes[key] for t in unseen)
            score = f'{passed}/{len(trials)} development scenes' if trials else 'Untested diagnostic'
            if trials and not passed:score=f'Failed ({passed}/{len(trials)})'
            result.append(dict(id=file.parent.name, label=f'{score} · {name} · {step:,} updates', checkpoint=checkpoint,
                               dataset=dataset,evaluation=evaluation,recommended=recommended,
                               full_trajectory=report.get('mode') == 'pilot_training', updated=file.stat().st_mtime))
        return sorted(result, key=lambda row: (row['recommended'],row['full_trajectory'],row['updated']), reverse=True)

    def start(self, candidate_id, seed):
        with self.lock:
            if self.running:
                raise ValueError('A VLA test is already running')
            if isinstance(seed, bool) or not isinstance(seed, int) or not 20000 <= seed < 30000:
                raise ValueError('VLA tests use development seeds 20000–29999')
            row = next((r for r in self.candidates() if r['id'] == candidate_id), None)
            if row is None or not row['recommended']:
                raise ValueError('A working visual policy is not available for this selection')
            python = self.root/'.venv-vla/bin/python'
            if not python.is_file():
                raise ValueError('Set up the separate VLA environment first; see docs/VLA.md')
            base = self.root/'outputs/vla/ui-runs'
            base.mkdir(parents=True, exist_ok=True)
            self.run_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            self.directory = base/self.run_id
            self.log = base/(self.run_id+'.log')
            self.stop_at = None
            args = [str(python), '-u', '-m', 'chora.vla.runner', '--checkpoint', str(row['checkpoint']),
                    '--dataset', str(row['dataset']), '--output', str(self.directory), '--seeds', str(seed),
                    '--max-ticks', '240', '--execute', '4', '--ui-progress']
            with self.log.open('wb') as log:
                self.process = subprocess.Popen(args, cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
            return self.run_id

    def stop(self):
        with self.lock:
            if self.running and self.stop_at is None:
                self.stop_at = time.monotonic()
                self.directory.with_suffix('.stop').touch()

    def reset(self):
        with self.lock:
            if self.running:
                raise ValueError("Stop the visual policy before resetting")
            self.process = self.directory = self.log = self.run_id = self.stop_at = None

    def status(self):
        with self.lock:
            if self.running and self.stop_at and time.monotonic()-self.stop_at > 5:
                self.process.terminate()
            candidates = self.candidates()
            chosen=next((r for r in candidates if r['recommended']),None)
            state = dict(ready=bool(chosen) and (self.root/'.venv-vla/bin/python').is_file(),
                         candidates=[dict(id=chosen['id'],label='Bowl placement visual policy',evaluation=chosen['evaluation'])] if chosen else [],
                         recommended_id=chosen['id'] if chosen else None,
                         running=self.running, id=self.run_id, phase='idle', has_frame=False)
            if not state['ready']:
                state['unavailable_reason'] = 'No visual policy has passed the development checks yet.'
            if not self.process:
                return state
            progress = read(self.directory/'progress.json', {})
            state.update(progress)
            state.update(running=self.running, id=self.run_id, has_frame=(self.directory/'latest.jpg').is_file())
            if self.running:
                state['phase'] = 'stopping' if self.stop_at else progress.get('phase', 'loading')
            else:
                report = read(self.directory/'report.json', {})
                trial = (report.get('trials') or [{}])[-1]
                state['phase'] = 'stopped' if self.stop_at or trial.get('reason') == 'stopped' else 'complete' if self.process.returncode == 0 else 'failed'
                state.update(outcome=trial.get('reason'), success=trial.get('success'),
                             report_url='/outputs/vla/ui-runs/'+self.run_id+'/report.json' if report else None)
                if state['phase'] == 'failed':
                    log = self.log.read_text(errors='replace')[-2000:] if self.log.exists() else ''
                    state['error'] = report.get('error') or (log.strip().splitlines()[-1] if log.strip() else 'VLA process exited before reporting a result')
            return state

    def frame(self):
        try:
            return (self.directory/'latest.jpg').read_bytes() if self.directory else b''
        except OSError:
            return b''

    def close(self):
        self.stop()
        if self.process and self.running:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
