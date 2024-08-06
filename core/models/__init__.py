from .pixart_video.pixart_video import PixArtVideo_XL_1x2x2, PixArtVideo_XL_1x1x1, PixArtVideo_g_1x2x2, PixArtVideo_dummy_1x2x2

from .text_encoder.clip import ClipEncoder
from .text_encoder.t5 import T5Encoder

model_cls = {
    'PixArtVideo_dummy_1x2x2': PixArtVideo_dummy_1x2x2,
    'PixArtVideo_XL_1x2x2': PixArtVideo_XL_1x2x2,
    'PixArtVideo_XL_1x1x1': PixArtVideo_XL_1x1x1,
    'PixArtVideo_g_1x2x2': PixArtVideo_g_1x2x2,
}