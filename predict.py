# Imports
import argparse
import utils
import json

parser = argparse.ArgumentParser(description='predict.py')

parser.add_argument('input', nargs='*', action="store", type=str, default='/home/workspace/aipnd-project/flowers/test/1/image_06743.jpg', help="File path of image to be predicted")
parser.add_argument('checkpoint', nargs='*', action="store", type=str, default='/home/workspace/aipnd-project/checkpoint.pth', help="File path of trained model")
parser.add_argument('--category_names', dest="category_names",  action="store", type=str, default='/home/workspace/aipnd-project/cat_to_name.json', help="File path of category names")
parser.add_argument('--gpu', dest="gpu", action="store", type=str, default="gpu", help="gpu or no_gpu, default is gpu")
parser.add_argument('--top_k', dest="top_k", action="store", type=int, default=5, help="Number of top predictions to be listed, default is 5")

args = parser.parse_args()
image_path = args.input
checkpoint_path = args.checkpoint
category_names = args.category_names
gpu = args.gpu
top_k = args.top_k

acceptable_gpu = ["gpu", "no_gpu"]
if gpu not in acceptable_gpu:
    print("Error message: Please use a valid gpu (gpu or no_gpu)")
    sys.exit()

if top_k < 1:
    print("Error message: Please select a number greater than 0 for top_k")
    sys.exit()

    
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

model = utils.load_checkpoint(checkpoint_path)

probs, classes = utils.predict(image_path, model, top_k, gpu)

name, labels = utils.get_labels(image_path, classes, cat_to_name)

print ("Actual class: {}".format(name))
print ("Model predicts:")
i = 0
while i < top_k:
    print ("{}. {} with a probability of {:.2f}%".format(i+1, labels[i], 100*probs[i]))
    i += 1