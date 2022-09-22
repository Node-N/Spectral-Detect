# Spectral-Detect
  
  
One component of the Spectral Detect project.  
  
  
Spectral Detect was developed in under 2 months as part of the University of Canterbury Professional Master of Computer Science degree which I worked on in a team of three. We worked with the Department of Conservation to prototype an application to detect Euphorbia paralias (sea spurge, highly invasive beach plant) in hyperspectral images. 
  
  
This module highlights solving a difficult problem. The hyperspectral images were up to 500 Gb and we had limited compute resources, so out of memory errors were common. I solved it by using the TensorFlow Dataset object, which allows the use of a generator to generate the tensor elements from the dataset (which is loaded as a memmap). 
Process.data_generator yields chunks of data after applying Factor Analysis, and when used to create a Dataset object we can map a function on the whole 500 Gb dataset!   
So we map the method Process.unfold_data which creates a window for each pixel, and then we can infer on each chunk and save the results.   
  
(the code could use a refactor, but we were under immense time pressure with many features to ship)  
  
  
  
The model, although not included, was adapted from [SpectralNET](https://github.com/tanmay-ty/SpectralNET  ).   
It is very interesting work, you should check it out.  
