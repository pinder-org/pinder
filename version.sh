#!/bin/bash

# determines new git tag from existing git tags
# and contents of the latest commit message which
# works just fine for squash + merge git workflows

# searches for "bumpversion command" where command is major|minor|patch|skip
# if command is skip, do not produce a new version, otherwise follow semver

TAGS=$(git for-each-ref --sort=creatordate --format='%(refname:short)' refs/tags)
LATEST_TAG=$(echo "${TAGS}" | grep -v "-" | tail -n 1 | awk -F' ' '{print $NF}')
LATEST_VERSION=${LATEST_TAG/v/}

# convert latest tag and HEAD to git commit hashes
LATEST_TAG_COMMIT=$(git rev-parse "${LATEST_TAG}")
HEAD_COMMIT=$(git rev-parse HEAD)

# if the head commit is already tagged with the latest tag, then we should not tag it again
if [[ "${LATEST_TAG_COMMIT}" == "${HEAD_COMMIT}" ]]; then
    exit 0
fi

if [[ "${LATEST_VERSION}" == "" ]]; then
    LATEST_VERSION=0.0.0
fi
BUMP=$(git log -n 1 | sed -n "/^.*bumpversion \(.*\)$/s//\1/p" | cut -f1 -d' ' | tail -n 1 | awk -F' ' '{print $NF}')
BUMP=$(python -c "print('${BUMP}'.lower() if '${BUMP}'.lower() in ['major', 'minor', 'patch', 'skip'] else 'patch')")
if [[ "${BUMP}" != "skip" ]]; then
    NEW_VERSION=$(python -c "import semver; print(semver.bump_${BUMP}('${LATEST_VERSION}'))")
    echo "v${NEW_VERSION}"
fi
