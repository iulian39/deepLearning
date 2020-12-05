# How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow (GPU)

## Brief Summary
Clone the full TensorFlow object detection repository located at https://github.com/tensorflow/models & extract all the files to models\research\object_detection
```
conda create -n tensorflow pip python=3.6
activate tensorflow
conda install -c anaconda protobuf
pip install --ignore-installed --upgrade tensorflow-gpu==1.15
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python scipy tf_slim pycocotools-windows lvis
```

Assumming the cloned repository is located at D:\tensorflow\
```
set PYTHONPATH=D:\tensorflow\models;D:\tensorflow\models\research;D:\tensorflow\models\research\slim
cd D:\tensorflow\models\research
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\flexible_grid_anchor_generator.proto .\object_detection\protos\calibration.proto ./object_detection/protos/center_net.proto ./object_detection/protos/fpn.proto
python setup.py build
python setup.py install
```

##Generate the TFRecord files, used for training:
```
python xml_to_csv.py # this will generate labels.csv files
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

##Train:
```
python train.py --logtostderr --train_dir=training_frcnn/ --pipeline_config_path=training_frcnn/faster_rcnn_inception_v2_pets.config
```

##Export Inference Graph:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training_frcnn/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training_frcnn/model.ckpt-XXXX --output_directory inference_graph
```
where “XXXX” in “model.ckpt-XXXX” located in training folder

##Eval:
```
python eval.py --logtostderr  --pipeline_config_path=training_frcnn/faster_rcnn_inception_v2_pets.config  --checkpoint_dir=inference_graph_frcnn/ --eval_dir=evals/
```
