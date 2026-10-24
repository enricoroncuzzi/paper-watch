# Changelog

## [0.7.1](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.7.0...paperwatch-v0.7.1) (2026-10-24)


### Bug Fixes

* mark runs that stored the Hugging Face listing ([#41](https://github.com/enricoroncuzzi/paper-watch/issues/41)) ([e917ea9](https://github.com/enricoroncuzzi/paper-watch/commit/e917ea9223d4a0b574c5af6cced7a35ad4635062))

## [0.7.0](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.6...paperwatch-v0.7.0) (2026-10-24)


### Features

* keep Hugging Face extras and record signal snapshots ([#37](https://github.com/enricoroncuzzi/paper-watch/issues/37)) ([385a2be](https://github.com/enricoroncuzzi/paper-watch/commit/385a2be412e6f936d8bb558a8b8cbec166f64ba7))


### Bug Fixes

* bump filelock to 4.2.4 and ignore two new unreachable Pillow advisories ([#38](https://github.com/enricoroncuzzi/paper-watch/issues/38)) ([4fb1e94](https://github.com/enricoroncuzzi/paper-watch/commit/4fb1e9432f0ae4345da4c331851de004f55fa6d3))

## [0.6.6](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.5...paperwatch-v0.6.6) (2026-10-23)


### Bug Fixes

* **sources:** drop credentials on cross-origin redirects, pin publicsuffixlist a week back, guard the drip test in CI ([#31](https://github.com/enricoroncuzzi/paper-watch/issues/31)) ([4c65cc6](https://github.com/enricoroncuzzi/paper-watch/commit/4c65cc62da54e80c5be0d6e91f7b9b26ef8dfa40))

## [0.6.5](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.4...paperwatch-v0.6.5) (2026-10-23)


### Bug Fixes

* **sources:** bound every fetch by size, time and redirect policy, and keep no transaction open across HTTP ([#29](https://github.com/enricoroncuzzi/paper-watch/issues/29)) ([11adb2c](https://github.com/enricoroncuzzi/paper-watch/commit/11adb2c047a45fd3e0cfcd3e6e5bcad8677e7f32))

## [0.6.4](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.3...paperwatch-v0.6.4) (2026-10-23)


### Documentation

* **deploy:** give the admin the pg_hba and pg_ident steps for the database roles ([#27](https://github.com/enricoroncuzzi/paper-watch/issues/27)) ([2dbc235](https://github.com/enricoroncuzzi/paper-watch/commit/2dbc2353daa58b0f357e01310610d70cac9fa6c8))

## [0.6.3](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.2...paperwatch-v0.6.3) (2026-10-23)


### Documentation

* **deploy:** give the API its own role, not the read-only one ([#25](https://github.com/enricoroncuzzi/paper-watch/issues/25)) ([7d8fa52](https://github.com/enricoroncuzzi/paper-watch/commit/7d8fa52503f28fdd495c9ec1811f087fcd4de6a7))

## [0.6.2](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.1...paperwatch-v0.6.2) (2026-10-23)


### Bug Fixes

* **deploy:** harden the morning check, the backup lock and retention, and loopback-only access ([#23](https://github.com/enricoroncuzzi/paper-watch/issues/23)) ([1708880](https://github.com/enricoroncuzzi/paper-watch/commit/1708880d4e096547675fbc731d2a63190705c465))

## [0.6.1](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.6.0...paperwatch-v0.6.1) (2026-10-23)


### Bug Fixes

* **cli:** resolve ingest config from the repo root and refresh pre-M3 docs ([#21](https://github.com/enricoroncuzzi/paper-watch/issues/21)) ([73dce1b](https://github.com/enricoroncuzzi/paper-watch/commit/73dce1b5b5b397b962ce306ceac74bf4a2d46944))

## [0.6.0](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.5.0...paperwatch-v0.6.0) (2026-10-23)


### Features

* **ingest:** retry Semantic Scholar refusals with follow-up passes ([#19](https://github.com/enricoroncuzzi/paper-watch/issues/19)) ([4595014](https://github.com/enricoroncuzzi/paper-watch/commit/4595014f9b9effb7e3ee010f257e2b6ee372b0a8))

## [0.5.0](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.4.1...paperwatch-v0.5.0) (2026-10-23)


### Features

* **sources:** tell Semantic Scholar refusals apart, stop after two refused chunks, and slack the re-check due time ([#17](https://github.com/enricoroncuzzi/paper-watch/issues/17)) ([0927538](https://github.com/enricoroncuzzi/paper-watch/commit/0927538836e4e15e618db5d5564a833778cbc349))


### Bug Fixes

* **ingest:** treat a first missed lab-feed day as a warning ([#16](https://github.com/enricoroncuzzi/paper-watch/issues/16)) ([e789dfe](https://github.com/enricoroncuzzi/paper-watch/commit/e789dfefb1e83bdfcede2c1068f460f2458fdd5b))

## [0.4.1](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.4.0...paperwatch-v0.4.1) (2026-10-22)


### Bug Fixes

* **sources:** join Semantic Scholar rows by arXiv id and record real run duration ([#14](https://github.com/enricoroncuzzi/paper-watch/issues/14)) ([687c7b5](https://github.com/enricoroncuzzi/paper-watch/commit/687c7b5404f7a127065bfafbb95242d70c41423b))

## [0.4.0](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.3.0...paperwatch-v0.4.0) (2026-10-22)


### Features

* **deploy:** run the daily ingestion on a shared VPS with systemd timers, backups and a morning check ([#12](https://github.com/enricoroncuzzi/paper-watch/issues/12)) ([9ce374c](https://github.com/enricoroncuzzi/paper-watch/commit/9ce374ca797131e8e1c6eec4ac510ed89ef3d317))

## [0.3.0](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.2.1...paperwatch-v0.3.0) (2026-10-21)


### Features

* **agent:** add question-answering agent, streaming API and QA eval ([#10](https://github.com/enricoroncuzzi/paper-watch/issues/10)) ([1b3e03c](https://github.com/enricoroncuzzi/paper-watch/commit/1b3e03cc4859457a1ca9cd4dcd7b5152d367a25a))

## [0.2.1](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.2.0...paperwatch-v0.2.1) (2026-10-19)


### Documentation

* add AGENTS.md and CLAUDE.md for coding agents ([#8](https://github.com/enricoroncuzzi/paper-watch/issues/8)) ([f697d41](https://github.com/enricoroncuzzi/paper-watch/commit/f697d419ae495e2000a0e5283b929199e393f96f))

## [0.2.0](https://github.com/enricoroncuzzi/paper-watch/compare/paperwatch-v0.1.0...paperwatch-v0.2.0) (2026-10-16)


### Features

* add tiered corpus ingestion and hybrid retrieval ([#5](https://github.com/enricoroncuzzi/paper-watch/issues/5)) ([35255dc](https://github.com/enricoroncuzzi/paper-watch/commit/35255dcad85c9785ae2d2fad8912fb97585a64be))


### Bug Fixes

* treat empty PW_* settings as unset ([c76b259](https://github.com/enricoroncuzzi/paper-watch/commit/c76b259a5373ad51728000186c712ee617573e5e))
