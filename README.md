# MAPPING THE SURFACE MESOSCALE OCEAN DYNAMICS IN THE SOUTH-WEST TROPICAL PACIFIC OCEAN USING SWOT ALTIMETRY DATA

## Context

The SWOT mission will enhance ocean dynamics observation to sub-mesoscale signals (<50 km) with its wide-swath measurement capability, providing 2 km resolution SSH data. At these fine scales, Sea Surface Height (SSH) variations are driven by Balanced Motions (BM) and Internal Tide (IT), requiring separation of these influences for effective use. The MEOM team at IGE, in collaboration with Datlas, is developing a method to separate BM and IT using a data assimilation system. The South-West Tropical Pacific, especially around French New-Caledonia, is a key study area due to strong IT generation. 

More information: [Master 2 Internship - Spring 2024](https://github.com/vbellemin/2024-internship-caledo) 

## Approach

In a nutshell, this intership project aims to develop an Observing System simulation Expeeriment (OSSE) for evaluating the effectiveness of Balanced Motion and Internal Tides reconstruction using spatially realistic Sea Surface Height (SSH) observations obtained by the SWOT and Nadir observations in the Southwestern Tropical Pacific around New Caledonia.

In this regard, the strategies taken by this challenge reflect both the scientific aspirations and the practical challenges encountered on the way that include:

- Adjust High-Resolution SSH observations from CALEDO60 dataset by incorporating atmospheric influences, and quantify the reliability of this adjusted dataset.
- Generate a Nature Run observations using the CALEDO60 dataset to simulate realistic ocean conditions and separate the balanced and unbalanced motions (BM-UM) from the high-resolution SSH data.
- Extract the BM component from the adjusted CALEDO60 dataset using a temporal high-pass filter to isolate the barotropic tide.
- Extract the Internal Gravity Waves from the adjusted CALEDO60 dataset using a spectral band filter designed to target the specific wave frequency.

This steps follow the next scheme:

![alt text](Images/nr_overview.png)

The internship was divided into two phases, reflecting the progression from foundational learning to practical application, which go as follows:

- Getting familiar with the New-Caledonia SSH simulation CALEDO60 and Training on data processing and plotting. 
- Developing an OSSE in New-Caledonia with particular emphasis on refining the generation of the Nature Run observations.

The work conducted so far has successfully led to the disentanglement of balanced and unbalanced motions through a series of sophisticated filtering techniques tailored to the CALEDO60 dataset. And adapting algorithms for coordinate interpolation, which had not been tested on this dataset before, posed significant challenges but is crucial for the implementation of the OSSE.

As the defense presentation is scheduled to conclude a month before the termination of this internship, there has been a strategic shift towards meticulously documenting the processes and results achieved so far. This approach ensures that the report simultaneously presents the methods and the results covered so far, facilitating clarity and highlighting the methodological aspect that is central to the objectives of the internship and the project itself.

By delving into the methodological intricacies, the document not only highlights the project's innovative contributions but also lays a solid groundwork for future students. Facilitating their entry into this dynamic line of research, presenting well-defined strategies and a robust foundation to further explore and expand upon.
