from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SegResNet, SwinUNETR

model = SegResNet(
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)

model = SwinUNETR(
    img_size=(192, 192, 96),
    in_channels=4,
    out_channels=3,
)

inference = SlidingWindowInferer(
    roi_size=(240, 240, 160),
    sw_batch_size=1,
    overlap=0.5,
)