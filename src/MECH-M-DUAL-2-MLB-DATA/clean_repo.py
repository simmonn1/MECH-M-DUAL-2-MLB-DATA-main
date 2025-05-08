import functools
from git import Repo, InvalidGitRepositoryError, GitCommandError

def require_clean_git(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            repo = Repo('.')
        except InvalidGitRepositoryError:
            raise RuntimeError("Not a valid Git repository.")

        if repo.is_dirty(untracked_files=True):
            raise RuntimeError("Repository has uncommitted changes. Please commit or stash them before proceeding.")

        try:
            repo.remotes.origin.fetch()
        except GitCommandError as e:
            raise RuntimeError(f"Failed to fetch from remote: {e}")

        local_commit = repo.head.commit
        remote_commit = repo.remotes.origin.refs[repo.active_branch.name].commit

        if local_commit.hexsha != remote_commit.hexsha:
            raise RuntimeError("Local branch is not up to date with remote. Please pull or push changes before proceeding.")

        return func(*args, **kwargs)

    return wrapper
