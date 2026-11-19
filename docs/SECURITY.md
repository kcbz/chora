# Local development security

The lab binds only to 127.0.0.1. Do not expose or tunnel it as a public service:
it has no account authentication and can start local simulation/training work.

Requests require a matching loopback Host header; supplied Origin headers must
match the lab. POST commands require bounded JSON-object bodies. Responses deny
framing to prevent clickjacking. These controls reduce browser-origin attacks;
they do not protect against another process already running as your OS user.

Keep `.env*`, private keys, model weights, datasets, caches and generated outputs
out of Git. `.env.example` is allowed only for placeholders. Do not put a real
credential in it. The repository does not require an OpenAI API key.

The pre-publication audit found no credential patterns in tracked files and no
OSV advisory matches for the 66 pinned package versions on 2026-11-19. This is a
point-in-time check, not a guarantee against unknown vulnerabilities or a complete
supply-chain audit. Recheck dependencies and secret exposure before publication.

If a credential is ever committed, revoke/rotate it and remove it from all
history to be published. Adding a .gitignore rule or a later deletion does not
remove earlier committed content.
