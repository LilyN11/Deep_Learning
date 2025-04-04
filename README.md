# Deep_Learning !!!DUE APRIL 14!!!
## Minimum required project (70%) [Pytorch]
Formulate a specific weak supervision problem to address and justify its usefulness and feasibility.
* Discuss and justify which weakly-supervised segmentation algorithm to use.
* Use references and citations to support your choice.
* Optionally, motivate, identify and collect additional weakly-labelled or unlabelled data.
* Implement a weakly-supervised segmentation framework.
* Design and conduct experiments for network comparisons, at least:
* Compare the framework with baseline models trained using fully-supervised methods

## To-do List 
- Everyone familiar with Github
- UCL Cluster SSH Access (Optional)

# Timeline
- **(Monday March 31st)**
  - Notes: Stephen implemented Resnet34 and U-net using the pixel level masks. 
    - Problems arising: trouble getting into UCL clusters and matplotlib working in the coursework environment **Ziad and Trevor will work with Stephen**
  - Rui implemented Grad Cam + CLIP 
    - Minor note: Clipseg is not weakly supervised (in between)
  - Task: Experiment with different versions of GradCam with a classifier
    - Affinitynet *Lily*
    - GradCam with Clip *Rui*
    - Bounding Boxes *Katherine and Hannah*
  - *Dani* will work on motivating, identifying and collecting additional weakly-labelled or unlabelled data
- **(Thursday April 3rd)** Discussion of results in progress 
- **(Tuesday April 8th)** Start Calculating Final Statistics for Models 
- **(Thursday April 10th)** Finish Experimentation 

## Groups:
* **Weakly-Supervised Team**: Rui, Lily, Hannah, and Katherine
  * *Rui* Segment Anything / Clip: https://github.com/facebookresearch/segment-anything *bounding box* 
  * *Hannah & Katherine* Grad Cam Plus: https://github.com/adityac94/Grad_CAM_plus_plus *In Tensorflow*
  * *Lily* Another Architecture/ Model 
* **Fully Supervised Team**: Ziad, Stephen, Dani, & Trevor 

## Tasks:
### Broader Questions
* Why weakly-supervised vs fully supervised (e.g. more efficient and less training data)
* From Slides:
  * Why the problem
  * Which algorithm, loss, optimiser, training strategy?
  * Does additional data help
  * metrics for segmentation tasks?
  * What are the experiments to compare these networks?
  * What can we conclude from the results?
### Methodlogical Tasks 
* (2) baseline fully supervised models (with pixel mask) 
* (2) baseline weakly supervised model
* Problem: Segmentation on the entire photo and not per pixel (without pixel mask)
* **Possible topics** to explore: image-level labels, bounding boxes, or partial annotations, multiple instance learning, class activation maps (CAMs), self-training, and consistency regularization
