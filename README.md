# virtual_media_task

Environment Configuration

As a systems student, I have very limited background in artificial intelligence. So first of all, I asked one of my friend who is studying AI, and got his recommandation, chosing 'CogVideo' as this task's target paper. 

Then attempted to configure the environment.

To simplify the installation process, I used Claude to read the official documentation and generate a machine-specific installation guide based on my hardware and operating system. The generated guide is included in this repository as configuration.pdf.

It should be noted that this README was written after I had already finished configuring the environment. Therefore, some details of the installation process were no longer completely clear at the time the document was written. In practice, the guide served as the main reference, but it was not sufficient by itself. During the installation, I still encountered several dependency and compatibility issues and had to install additional packages and resolve configuration problems manually. Consequently, the environment was built based primarily on this guide, with several additional adjustments made during the actual setup.

Another practical consideration concerns the model version. The paper analyzed in this report is the original CogVideo paper published in 2023. However, according to the official documentation, the original implementation requires relatively demanding hardware resources and is primarily designed for Chinese-language prompts. Since my experimental platform only provides an NVIDIA RTX 3090 (24 GB VRAM), I instead used CogVideoX, a newer version of the project that is better optimized for current hardware. In addition, CogVideoX also supports English and even Japanese prompts.

Therefore, although this report discusses the original CogVideo paper, the experiments and demonstrations presented in the following sections were conducted using the CogVideoX implementation.

Since this assignment required reproducing and understanding the CogVideo project, I first familiarized myself with the paper and then attempted to configure the runtime environment.
