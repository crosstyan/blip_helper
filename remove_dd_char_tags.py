import os

pretrained_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pretrained")
dd_dir = os.path.join(pretrained_path, "deepbooru")

general_tags_path = os.path.join(dd_dir, "tags-general.txt")
tag_path = os.path.join(dd_dir, "tags.txt")
if general_tags_path.exists() and tag_path.exists():
  os.remove(tag_path)
  os.renames(general_tags_path, tag_path)
  print("Removed tags-general.txt and renamed tags-general.txt to tags.txt")
else:
  print("no tags-general.txt and tags.txt found")
