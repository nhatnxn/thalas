1. Clone the code
2. install requirements: 
`$pip install -r requirements.txt`


3. Download pretrained models and put in the folder `TabDetectDewarp`. The folder structure would be:

    ```bash
    |____TabDetectDewarp
    | |____tab_seg.py
    | |____tab_det.py
    | |____yolov5
    | | |____.gitignore
    | | |____infer.py
    | | |____models
    | | |____requirements.txt
    | | |____test.ipynb
    | | |____utils
    | | |____weights
    | | | |____yolov5s_thalas.pt                <========     Put downloaded pretrained model here
    | | |______init__.py
    |____inference.py
    |____samples
    |____.gitignore
    |____requirements.txt
    |____readme.md
    ```