#!/bin/sh
# Vercel runs this before a build: exit 1 means build, exit 0 means skip.
#
# Only a release commit builds, and only in production. The page then changes together with the
# server release, and no preview deployment (an unprotected URL) is ever created.
if [ "$VERCEL_ENV" != "production" ]; then
  exit 0
fi

case "$VERCEL_GIT_COMMIT_MESSAGE" in
  "chore(main): release"*) exit 1 ;;
  *) exit 0 ;;
esac
