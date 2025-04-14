# Beginner Projects (As of 4/13/25)

#  1. Custom Image Classifier (from scratch)

**Goal:** Train your own CNN (or fine-tune a known one) on a custom image dataset.

- **Use:** TensorFlow/Keras or PyTorch  
- **Dataset:** Kaggle, your phone camera, or scrape your own with Python  

**Add your own tweaks:**
- Try `1×1` convolutions or replace `5×5` with two `3×3`s  
- Use `Dropout + BatchNorm` combos  
- Replace the final FC layers with **Global Average Pooling**  

**Examples:**
- Classify different types of **flowers, food, car models, or Pokémon**  
- Fine-tune **InceptionV3** on a smaller, custom dataset  

---

#  2. Feature Visualization Project

**Goal:** Visualize activations and filters at various layers of a CNN

- Helps solidify your understanding of what each convolutional layer learns  
- You can create a dashboard that shows:
  - Filters  
  - Intermediate feature maps  
  - Class Activation Maps (CAMs)  

---

#  3. Object Detection (YOLO-lite style)

**Goal:** Implement a simple object detection model using bounding boxes.

- Start with a dataset like **Pascal VOC** or custom-label a small one using **LabelImg**  
- You’ll learn:
  - Multi-label regression  
  - Intersection over Union (IoU)  
  - Anchor boxes (eventually)  

---

#  4. Style Transfer or Image-to-Image Translation

**Goal:** Use CNNs for creativity!

- Transfer the style of one image (like **Van Gogh**) onto another  
- Use the **encoder-decoder** structure and modify it with your knowledge of conv layers and pooling  

---

#  5. Image Noise Reduction with Autoencoders (Currently working on this)

**Goal:** Build a convolutional autoencoder to remove noise from images.

- Apply **Gaussian noise** or **salt & pepper noise** to an image  
- Train the model to clean it  
- Great for applying convolutional theory + understanding of **MSE loss**  

---

# Bonus: Compare Architectures on the Same Dataset

- Take **CIFAR-10** or **MNIST**  
- Train the same dataset using:
  - **VGG19**  
  - **AlexNet**  
  - **Your own lightweight CNN**  

- Log performance differences:
  - Accuracy  
  - Training time  
  - Model size  

- Use **Global Average Pooling** in one version, **FC layers** in another  
