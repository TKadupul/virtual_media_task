# virtual_media_task

As a systems student, I have very limited background in artificial intelligence. So first of all, I asked one of my friend who is studying AI, and got his recommandation, chosing 'CogVideo' as this task's target paper. 

Then attempted to configure the environment.

Section 1 Configuration
To simplify the installation process, I used Claude to read the official documentation and generate a machine-specific installation guide based on my hardware and operating system. The generated guide is included in this repository as configuration.pdf.

It should be noted that this README was written after I had already finished configuring the environment. Therefore, some details of the installation process were no longer completely clear at the time the document was written. In practice, the guide served as the main reference, but it was not sufficient by itself. During the installation, I still encountered several dependency and compatibility issues and had to install additional packages and resolve configuration problems manually. Consequently, the environment was built based primarily on this guide, with several additional adjustments made during the actual setup.

Another practical consideration concerns the model version. The paper analyzed in this report is the original CogVideo paper published in 2023. However, according to the official documentation, the original implementation requires relatively demanding hardware resources and is primarily designed for Chinese-language prompts. Since my experimental platform only provides an NVIDIA RTX 3090 (24 GB VRAM), I instead used CogVideoX, a newer version of the project that is better optimized for current hardware. In addition, CogVideoX also supports English and even Japanese prompts.(models--THUDM--CogVideoX-5b)

Therefore, although this report discusses the original CogVideo paper, the experiments and demonstrations presented in the following sections were conducted using the CogVideoX implementation.

Since this assignment required reproducing and understanding the CogVideo project, I first familiarized myself with the paper and then attempted to configure the runtime environment.

OK, the environment is OK
Let's start to generate the first video:
use the demo command to generate a little boy running in the rain
    --prompt "A small boy, head bowed and determination etched on his face, sprints through the torrential downpour as lightning crackles and thunder rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the dramatic sky's anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements."  --model_path THUDM/CogVideoX-5b   --generate_type t2v  --num_frames 49  --output_path ./little_boy.mp4

as you can see, the first video is in:
    CogVideo/my_first_video.mp4 (truly first video)
    CogVideo/little_boy.mp4(example)



Section 2 Reading

Although I have read almost no AI papers, I would like to summarize my understanding of this paper based on my limited knowledge.

At the time, text-to-image generation was already a relatively mature technology, while text-to-video generation was still underdeveloped.

The strategy adopted in this paper is somewhat like teaching someone who already knows how to draw how to make an animation. The authors use the mature CogView2 model to generate individual images, while the main contribution of this paper is to teach the model how to make those images move.

Before this work, text-to-video methods faced several problems.

As video generation progressed, the visual content would gradually drift away from the original text prompt. Vanilla autoregressive models were relatively good at generating regular motion, such as a car continuously moving forward, or random patterns, such as a person speaking with randomly moving lips.

However, they struggled with videos such as “a lion is drinking water,” because this action requires a precise understanding of a temporally ordered process:

The lion holds a glass;

The lion raises the glass to its mouth;

The lion drinks the water;

The lion puts the glass down.

The model must capture the meaning and order of the action “drinking,” rather than merely generating random visual changes.

By contrast, in a video of “a car moving forward,” the first frame often already contains sufficient information. The model only needs to continue moving the car forward.

One reason for this problem is that previous methods usually divided a video into many short clips with a fixed number of frames and then used those clips for training. This treatment could break the correspondence between the text and the temporal process represented in the video.

For example, a complete “drinking” video might contain the following stages:

Holding a glass;

Lifting the glass;

Drinking;

Putting the glass down.

If the video were divided into four short clips, while all four clips were assigned the same caption, “drinking,” the model could become confused:

Does “drinking” refer to holding the glass, lifting it, actually drinking, or putting it down?

In other words, the model would observe only local parts of the action without knowing where those parts belonged in the overall process.

Previous text-to-video methods generally used a fixed frame rate. This paper proposes a different strategy.


generation process.
First of all

Input text:

A lion is drinking water.

During training, the video frames are first processed by an image tokenizer and converted into visual tokens.

Stage 1: Sequential Generation

In this stage, the model generates five key frames at a low frame rate. Roughly speaking, these frames are sampled across the temporal span of the video so that they cover the overall action. （split it into five equal parts）

Stage 2: Recursive Interpolation

The key frames generated at a low frame rate may have large jumps between them.

For example:

The lion has not yet started drinking
→ The lion has suddenly finished drinking

The second stage therefore inserts intermediate frames between the key frames. The frame rate is increased recursively, with additional frames inserted at each round.

To support interpolation, CogVideo uses bidirectional attention. In ordinary autoregressive generation, when the model generates the current frame, it can only see the content that has already been generated. In an interpolation task, however, the model needs to refer to both the known frame before the missing position and the known frame after it.

Therefore, CogVideo assigns some tokens to bidirectional attention regions, allowing the model to use information from both sides of the interpolation position.

There is another important design choice.

We want the person who already knows how to draw to learn how to make animations, without having to retrain their entire drawing ability from scratch. CogVideo therefore uses dual-channel attention.

The spatial channel represents the model’s original ability to generate images, inherited from the text-to-image model. This part is kept frozen.

The temporal channel is the newly trained component. It is mainly responsible for learning how different frames change over time.

summary

CogVideo first generates a small number of key frames according to the text prompt, then recursively inserts intermediate frames between them. At the same time, two cooperating attention channels within the model handle image content and temporal changes, respectively.

Overall Pipeline
    Text prompt: A lion is drinking water --> Add frame-rate information
    During training, encode video frames with VQ-VAE into visual tokens
    Stage 1: Sequentially generate five low-frame-rate key frames
    Stage 2: Recursively insert intermediate frames
    Obtain a temporally coherent video


