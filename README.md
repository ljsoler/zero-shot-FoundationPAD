#### This is the official repository of the paper:
### Are Foundation Models All You Need for Zero-shot Face Presentation Attack Detection?
### Paper accepted at [Face & Gesture 2025]([https://www.vislab.ucr.edu/Biometrics2024/index.php](https://fg2025.ieee-biometrics.org/))

## Requierements ##
- Python 3.8+
- pytorch-lightning==2.1.0
- torch==2.1.0
- pyeer

<hr/>

## Zero-shot foundation model training ##

- python run.py --config configs/config_cross_db.yaml

<hr/>

## Foundation model testing ##
- python run_test.py --config configs/config_cross_db.yaml --ap_folder <path-to-attacks-folder> --bp_folder <path-to-bonafide-folder> --model_weights <path-to-model-weights> --output_folder <output-folder-to-save-metrics>

<hr/>

## Unknown PAI species performance ##

![Unknown PAI species performance](/img/SiW-Mv2.png)  
*Tab III: Detection performance (in \%) of foundation models for the SiW-Mv2 leave-one-out protocol. The best overall results are highlighted in bold.*


## Cross-database performance ##

![Cross-database performance](/img/cross-database.png)  
*Tab IV: Benchmark (in %) of foundation models against the state of the art for different cross-database settings. The best results are highlighted in bold.*

<hr/>

## Citation
```
@InProceedings{GonzalezSoler_ZeroFoundPAD_FG_2025,
    author    = {L. J. Gonzalez-Soler, J. E. Tapia, C. Busch},
    title     = {Are Foundation Models All You Need for Zero-shot Face Presentation Attack Detection?},
    booktitle = {Proc. Intl. Conf. on Automatic Face and Gesture Recognition (FG)},
    month     = {May},
    year      = {2025},
    pages     = {}
}
```

```
@InProceedings{Tapia_ZeroIrisFoundPAD_FG_2025,
    author    = {J. E. Tapia, L. J. Gonzalez-Soler, C. Busch},
    title     = {Towards Iris Presentation Attack Detection with Foundation Models},
    booktitle = {Proc. Intl. Conf. on Automatic Face and Gesture Recognition (FG)},
    month     = {May},
    year      = {2025},
    pages     = {}
}
```

## License
>This project is licensed under the terms of the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.  
Copyright (c) 2025 Hochschule Darmstadt  
For more details, please take a look at the [LICENSE](./LICENSE) file.


