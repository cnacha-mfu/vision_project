# Vision Project
some sample code for YOLO and many more

pip install ultralytics
pip install opencv-python

## Install ByteTrack
for tracking object real-time 

```
%cd {HOME}
git clone https://github.com/ifzhang/ByteTrack.git
%cd {HOME}/ByteTrack

# workaround related to https://github.com/roboflow/notebooks/issues/80
sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt

pip3 install -q -r requirements.txt
python3 setup.py -q develop
pip install -q cython_bbox
pip install -q onemetric
# workaround related to https://github.com/roboflow/notebooks/issues/112 and https://github.com/roboflow/notebooks/issues/106
pip install -q loguru lap thop
```
