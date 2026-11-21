"""Names shared by the measuring and the deciding code, with no heavy imports.

The early-exit network has exits after `layer2`, `layer3` and `layer4`, named
after the ResNet block each follows, so a row says which exit it describes
rather than an index. Every exit but the last is also a point where the
network can be split and its feature map sent.
"""

EXIT_NAMES = ("layer2", "layer3", "layer4")
SPLIT_NAMES = EXIT_NAMES[:-1]
