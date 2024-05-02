# NOISe_Whole_Slide_Inference_InProgress


 Development of whole-slide inference for the NOISe model.
 
 Images too high-resolution to be processed by the model will be divided into overlapping patches where each patch is processed individually. The resulting predictions of each patch will then be compared to their neighboring patches; in a modified form of non-maxmima suppression, predictions with sufficiently overlapping bounding boxes will be removed as duplicates. The remaining predictions are then collated and scaled to the original whole-slide image.

