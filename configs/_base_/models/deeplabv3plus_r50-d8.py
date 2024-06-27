# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    # decode_head=dict(
    #     # type='SegFormerHead',
    #     # type='FeedFormerHead_mit',
    #     type='FeedFormerHead_new',
    #     mlp_ratio=4,
    #     # type='FeedFormerHead_ori',
    #     in_channels=[256, 512, 1024, 2048],
    #     # in_channels=[192, 192, 192, 192],
    #     # ham_channels=256,
    #     in_index=[0, 1, 2, 3],
    #     feature_strides=[4, 8, 16, 32],
    #     channels=128,
    #     dropout_ratio=0.1,
    #     num_classes=150,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     # decoder_params=dict(embed_dim=256),
    #     decoder_params=dict(embed_dim=512),
    #     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
