## ⭐ 🌉 Golden Gate CLIP ✨🤖 🥳 🌉
----

Inspired by [Anthrophic's Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude), this repo obtains CLIP Vision Transformer feature activations for a set of images, compares which feature indices are present in all images, and then manipulates the activation value for that neuron in the CLIP Vision Transformer MLP c_fc (Fully Connected Layer).

Result: CLIP predicts "San Francisco, Bay Area, sf, sfo" even for an image that it otherwise (rightfully!) describes as being noise.

![CLIP-golden-gate](https://github.com/zer0int/Golden-Gate-CLIP/assets/132047210/bd18afa8-e220-4d9f-aa50-2d36a045d8b0)

- To reproduce: python run_clipga-goldengate-manipulate-neuron-activation.py --image_path noise/ggnoise.png
- ...And run the files starting with "i-" as-is to get the activation values I used in above code.

To make your own:

- Edit the folder name in the two files starting with "i-" to be e.g. photos of your cat.
- Resulting text files show you which feature numbers (indices) are present in all images.
- This helps to distinguish between vases, sofas, and other things not always around your cat from your actual cat.
- If none, the "cat" might not be salient in some images. Check the "-all" files manually**!
- 
- ** I intend to make this easier in the future, by displaying outliers so you can remove them.
- For now, .json / .csv can be opened with a text editor and manually CTRL+F compared.
----
- For best results, manipulate Layer 21, 22 features (penultimate and 3rd layer near the output).
- Features <=~5 (input) encode simple lines and zigzags --> near output: Multimodal complex features.
- Check "run_clipga-", I explained how to manipulate activations in the #code #comments!

🫤 Limitation: Needs at least somewhat similar images (e.g. Golden Gate Bridge <-> Any random bridge) to work when manipulating the activation value on a single layer. You'd likely have to coherently trick CLIP into the right activations over multiple layers to make it totally obsessed about the Golden Gate Bridge like GG Claude was. However, this toolkit is a great start!

⚠️ *Files starting with "x-" allow you to obtain absolutely every activation value in CLIP, including Multihead Attention, projection layer, individual patches (image tokens) + CLS. If you have a use for them, I am assuming you'll need no introduction on how to use them. Enjoy!

----
Original CLIP Gradient Ascent Script: Used with permission by Twitter / X: [@advadnoun](https://twitter.com/advadnoun)
- GG Bridge images: Via Google & Google Image Search, downsized to minimum resolution 336x336 pixels, limited to 5 (fair use).
