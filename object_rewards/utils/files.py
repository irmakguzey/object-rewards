import git


def get_root_path():
    repo = git.Repo(".", search_parent_directories=True)
    return repo.working_tree_dir
