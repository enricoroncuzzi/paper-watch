# 1. Record architecture decisions

Date: 2026-10-11

## Status

Accepted

## Context

Paper Watch makes decisions that are expensive to reverse, such as the database, the cloud
provider, the agent framework, and how sources are trusted. Without a record, the reasons behind
them are lost, and later changes either repeat old debates or break assumptions nobody remembers.

## Decision

We record every architecturally significant decision as an Architecture Decision Record (ADR) in
`docs/adr/`, following Michael Nygard's format: Title, Status, Context, Decision, Consequences.
ADRs are numbered sequentially and never deleted; a reversed decision gets a new ADR that marks
the old one as superseded.

## Consequences

- Anyone reading the code can find why a structural choice was made, not only what it is.
- Writing an ADR is part of the change that makes the decision, reviewed in the same pull request.
- Small, easily reversible choices do not need an ADR.
