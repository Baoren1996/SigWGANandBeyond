from lib.algos.sigcwgan import SigCWGANConfig
from lib.augmentations import get_standard_augmentation, SignatureConfig, Scale, Concat, Cumsum, AddLags, LeadLag, Addtime

SIGCWGAN_CONFIGS = dict(
    ECG=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.05)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.05)),
    ),
    VAR1=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag(with_time=False)])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    VAR2=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR3=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.5), Cumsum(), AddLags(m=2), LeadLag(with_time=True)])),
        sig_config_future=SignatureConfig(depth=2, augmentations=tuple([Scale(0.5), Cumsum(), AddLags(m=2), LeadLag(with_time=True)])),
    ),
    VAR10=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR20=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR50=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR100=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    STOCKS_SPX=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag(with_time=True)])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag(with_time=True)])),
    ),
    STOCKS_SPX_DJI=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag(with_time=True)])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag(with_time=True)])),
    ),
    ARCH=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=3, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=3, augmentations=get_standard_augmentation(0.2)),
    ),
)



MCSWGAN_CONFIGS = dict(
    ECG=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True)),
    VAR1=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True)),
    VAR2=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True)),
    VAR3=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=3, augmentations=(Scale(0.8),Cumsum(),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=3, augmentations=(Scale(0.8),Cumsum(),Addtime()),basepoint=True)),
    VAR10=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR20=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR50=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR100=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    STOCKS_SPX=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True)),
    STOCKS_SPX_DJI=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True)),
    ARCH=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5),Addtime()),basepoint=True)),
)


