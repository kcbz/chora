from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Chora by Coherence — local robot learning")
    parser.add_argument("--experiment",help="Experiment preset ID; defaults to so101-bowl")
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="Check the isolated environment and assets")
    p = sub.add_parser("expert",help="Check the scripted expert")
    p.add_argument("--episodes",type=int,default=50)
    p.add_argument("--live",action="store_true")
    p.add_argument("--record",action="store_true")
    p.add_argument("--seed",type=int,default=0)
    p = sub.add_parser("generate",help="Generate local expert demonstrations")
    p.add_argument("--episodes",type=int)
    sub.add_parser("curate",help="Filter, balance, and split demonstrations")
    sub.add_parser("download",help="Download the 4-bit Qwen model (~1 GB)")
    sub.add_parser("report",help="Rebuild the report and evaluation gallery from measured outputs")
    p = sub.add_parser("train",help="Train a local QLoRA adapter using MLX")
    p.add_argument("--iterations",type=int)
    p.add_argument("--batch-size",type=int)
    p.add_argument("--resume",action="store_true")
    p = sub.add_parser("evaluate",help="Evaluate on reserved seeds")
    p.add_argument("--policy",choices=["expert","base","student"],default="student")
    p.add_argument("--episodes",type=int)
    p.add_argument("--record",type=int,default=3)
    p = sub.add_parser("preview",help="Preview an immutable checkpoint on development scenes")
    p.add_argument("--checkpoint",type=int)
    p.add_argument("--episodes",type=int,default=5)
    p = sub.add_parser("select",help="Freeze a checkpoint before the final benchmark")
    p.add_argument("--checkpoint",type=int,required=True)
    p = sub.add_parser("replay",help="Re-render benchmark clips with verified identical outcomes")
    p.add_argument("--policy",choices=["expert","base","student"],default="student")
    p.add_argument("--failures",type=int,default=2)
    p = sub.add_parser("thumbnails",help="Render verified final-state thumbnails for scored episodes")
    p.add_argument("--policy",choices=["all","expert","base","student"],default="all")
    p = sub.add_parser("serve",help="Open the local robot learning lab")
    p.add_argument("--port",type=int,default=8765)
    p = sub.add_parser("vla-record",help="Record fixed-tick visual expert pilot episodes")
    p.add_argument("--episodes",type=int,default=3)
    p.add_argument("--seed",type=int,default=20000)
    p.add_argument("--output",help="New run directory; existing directories are never overwritten")
    p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--state-only",action="store_true",help="Physics diagnostic only; excludes training images")
    p = sub.add_parser("vla-replay",help="Verify a tick recording against its full physics trajectory")
    p.add_argument("episode",help="Path to an episode NPZ file")
    args = parser.parse_args()
    if args.experiment:
        import os
        os.environ["CHORA_EXPERIMENT"]=args.experiment
    from .common import config
    if config().get("physics",{}).get("learning_enabled") is False and args.command in {"generate","curate","train","evaluate","preview","select","thumbnails","replay"}:
        parser.error("This deformable-material trial is not a validated learning benchmark. Run it in the lab; scoring/training is unavailable.")
    if args.command=="expert" and config().get("physics",{}).get("learning_enabled") is False and not (args.live or args.record):
        parser.error("This is a material trial, not a scored expert check. Use the lab or expert --record.")
    if args.command=="vla-record":
        from .vla.record import collect
        print(collect(episodes=args.episodes,seed_start=args.seed,output=args.output,
                      images=not args.state_only,image_size=args.image_size))
    elif args.command=="vla-replay":
        from .vla.record import replay
        print(json.dumps(replay(args.episode),indent=2))
    elif args.command=="doctor":
        import importlib.metadata
        import platform
        import sys
        from .common import ROOT, OUTPUT, config
        for name in ["mujoco","numpy","mlx","mlx-lm"]:
            try: version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: version = "not installed"
            print(f"{name}: {version}")
        print(f"Python: {sys.version.split()[0]} / {platform.machine()}")
        print(f"Environment: {sys.prefix}")
        print(f"Experiment: {config()['experiment']['name']}")
        print(f"Outputs: {OUTPUT}")
        print(f"SO-101 assets: {(ROOT/'assets/menagerie/robotstudio_so101/so101.xml').exists()}")
    elif args.command=="expert":
        from .episodes import oracle_check
        if args.live or args.record:
            from .sim import make_simulation
            from .policy import make_expert
            from .visualization import Recorder, watch_live
            from .common import OUTPUT
            sim = make_simulation(); sim.reset(args.seed)
            policy = make_expert(args.seed)
            if args.live:
                watch_live(sim,policy)
            else:
                r = Recorder(sim,OUTPUT/"expert.mp4")
                history=[]
                try:
                    while not sim.finished:
                        action = policy.act(sim.observe(),history)
                        sim.step(action); history.append(action)
                finally: r.close()
                print(f"Recording: {OUTPUT/'expert.mp4'} / success: {sim.success}")
        else:
            oracle_check(args.episodes)
    elif args.command=="generate":
        from .episodes import generate
        generate(args.episodes)
    elif args.command=="curate":
        from .episodes import curate
        curate()
    elif args.command=="download":
        from .learn import download_model
        download_model()
    elif args.command=="report":
        from .report import build_report
        print(build_report())
    elif args.command=="train":
        from .learn import train
        train(args.iterations,args.resume,args.batch_size)
    elif args.command=="evaluate":
        from .learn import evaluate
        print(json.dumps(evaluate(args.episodes,args.policy,args.record),indent=2))
    elif args.command=="preview":
        from .learn import preview
        print(json.dumps(preview(args.checkpoint,args.episodes),indent=2))
    elif args.command=="select":
        from .learn import select_checkpoint
        print(json.dumps(select_checkpoint(args.checkpoint),indent=2))
    elif args.command=="replay":
        from .learn import replay
        replay(args.policy,args.failures)
    elif args.command=="thumbnails":
        from .gallery import thumbnails
        thumbnails(args.policy)
    elif args.command=="serve":
        from .server import serve
        serve(args.port)
