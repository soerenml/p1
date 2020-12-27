version='5'

# Run the model training
python task.py \
  --steps_epoch 10 \
  --epochs 200 \
  --batch_size 20 \
  --data_path '/Users/soeren/data/cats_and_dogs_filtered' \
  --model 'model_1' \
  --run_id $version

# Start Tensorboard
tensorboard --logdir '/Users/soeren/tf-logs/'${version}
