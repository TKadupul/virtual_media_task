# virtual_media_task

As a systems student, I have very limited background in artificial intelligence. So first of all, I asked one of my friend who is studying AI, and got his recommandation, chosing 'CogVideo' as this task's target paper. 

Then attempted to configure the environment.

Section 1 Configuration
To simplify the installation process, I used Claude to read the official documentation and generate a machine-specific installation guide based on my hardware and operating system. The generated guide is included in this repository as configuration.pdf.

It should be noted that this README was written after I had already finished configuring the environment. Therefore, some details of the installation process were no longer completely clear at the time the document was written. In practice, the guide served as the main reference, but it was not sufficient by itself. During the installation, I still encountered several dependency and compatibility issues and had to install additional packages and resolve configuration problems manually. Consequently, the environment was built based primarily on this guide, with several additional adjustments made during the actual setup.

Since this assignment required reproducing and understanding the CogVideo project, I first familiarized myself with the paper and then attempted to configure the runtime environment.

## Execution Environment

- Operating system: Rocky Linux 9.8 (Blue Onyx), x86_64
- System Python: Python 3.9.25
- GPU: NVIDIA GeForce RTX 3090
- GPU memory: 24,576 MiB
- NVIDIA driver: 610.43.02
- Model: THUDM/CogVideo-5b
- Inference precision: FP16

A representative inference command was:

```bash
python inference/cli_demo.py \
  --prompt "PROMPT" \
  --model_path THUDM/CogVideo-5b \
  --generate_type t2v \
  --num_frames 49 \
  --num_inference_steps 50 \
  --guidance_scale 6.0 \
  --fps 8 \
  --dtype float16 \
  --output_path ./tests/test.mp4
```
The guidance scale was generally set to 6.0 to balance prompt adherence and visual quality. The number of inference steps was fixed at 50.

Approximate generation times were:

- 49 frames: 10–15 minutes
- 81 frames: 20–30 minutes or longer
- 161 frames: approximately 60–90 minutes

The number of frames and inference steps can be reduced when faster generation is required, although this may reduce the generation quality.





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

Overall process
    Text prompt: A lion is drinking water --> Add frame-rate information
    During training, encode video frames with VQ-VAE into visual tokens
    Stage 1: Sequentially generate five low-frame-rate key frames
    Stage 2: Recursively insert intermediate frames
    Obtain a temporally coherent video



Section 3:
Failure case

Before presenting the failure analysis, it is important to note that the tested model exhibited limited generation quality and robustness. Many generated videos contained noticeable visual artifacts, prompt-following errors, and temporal inconsistencies. Therefore, a relatively broad range of failure cases could be observed.

I first reviewed the example failure cases provided in the assignment instructions, designed corresponding test prompts, and experimentally evaluated them which could be completed within the available computational and time constraints.
identification \ hallucination \ temporal inconsistance \ bias

1, identification
### Failure on Specific Objects or under Specific Conditions

I consider **“misrecognition under specific conditions”** and **“failure only for specific objects”** to be broadly similar problems in this experiment, so I grouped them into the same category.

Originally, I wanted to generate a scene in which a knight fights a monster commonly found in a video game. However, even when I used the general word “monster,” the model failed to generate a corresponding monster.

In **Test 1**, I asked the model to generate a battle between a knight and a monster. Since the monster was not generated correctly, I conducted **Test 2**, which showed two knights fighting, as a comparison.

Similarly, I wanted to use relatively uncommon underwater creatures to test whether the model would fail when generating objects from a more specialized domain.

- **Test 3:** An underwater diving scene  
- **Test 4:** Two sperm whales  
- **Test 5:** A deep-sea fish  
- **Test 6:** A seahorse  

For Test 5, the term used in the prompt was not sufficiently accurate. I intended to generate a frightening deep-sea anglerfish, but failed. 

Surprisingly, the underwater scenes in Tests 3,4, and 6 were generated relatively well. Therefore, in **Test 7**, I tested a more unusual fictional creature—Cthulhu. This time, the model failed to generate the requested object correctly.


2, hallucination
### Hallucination

To test hallucination, I deliberately added the instruction **“No other objects or people appear”** to each prompt. However, all three tests failed to follow this restriction.

In **Test 1**, I asked the model to generate one apple placed on a table. However, the generated video contained many apples instead of only one.

In **Test 2**, I requested one red cube and one blue ball. Although the overall quality of the generated video was quite good, an additional human hand appeared, even though the prompt explicitly stated that no people should appear.

In **Test 3**, I asked the model to generate a blue cup. However, a chair appeared at the beginning of the video.

3, temporal inconsistance
### 3. Temporal Inconsistency

The original plan for this experiment was not successfully completed. I initially prepared two examples to test temporal inconsistency.

In the first example, I wanted to generate a video in which one car overtakes another car and then makes a U-turn. I expected that after the car turned around, the appearance of the car that had previously been overtaken might change.

In the second example, I wanted to generate a video in which the camera completed a full rotation. I expected that after the 360-degree rotation, the surrounding scenery would be different from what appeared at the beginning.

However, the model appeared to be too limited to generate the actions required by these two planned experiments.

#### Car U-turn experiment

Tests 1 and 6 were both related to the car example. The results suggest that the model could not understand the action of a car making a U-turn. It completely failed to simulate this action. I attempted the experiment several times, but none of the generated videos successfully showed the requested U-turn.

#### Camera rotation experiment

All the remaining test videos were related to the camera-rotation example.

At first, I used the default setting of 49 frames. Unfortunately, even after several attempts, the camera could not complete a full rotation. Since the generated video was only approximately six seconds long, the camera appeared unable to finish a complete 360-degree rotation within the available duration. I kept only Test 2 as a demonstration of this result.

I then increased the number of frames to 81, producing a video of approximately ten seconds. Except for Test 5, the remaining videos in this experiment used this ten-second setting.

In Test 3, for example, the camera successfully completed one full rotation. An interesting behavior of the model was that it appeared to fix the first and last frames so that they were visually consistent. Therefore, even after a 360-degree rotation, the ending frame was similar to the starting frame. In a way, the result could be considered temporally consistent.

I therefore began another experiment in which I asked the model to complete two full rotations. However, similar to the previous results, the camera appeared able to complete only one rotation within ten seconds. I repeated the experiment many times and changed the prompt in different ways, but the camera still completed only one rotation—or sometimes less of it.

I then increased the video length again, this time to 161 frames. In this experiment, the camera rotated many times, even though the prompt requested only two rotations. As shown in Test 5, the video also became almost impossible to recognize, and the overall generation quality was extremely poor.

Because each 161-frame generation required a very long running time, I tested this setting only three or four times. The results were consistently similar to Test 5, so I did not continue testing it.

Therefore, the original temporal-consistency test could not be completed as planned. Nevertheless, these unsuccessful experiments revealed several other problems.

For example, in Test 7, the prompt requested a 720-degree camera movement around a statue. During the rotation, the surrounding scenery changed, but the statue appeared to move together with the camera. This created a very unnatural sense of space. Instead of showing the statue continuously from different viewing angles, the statue itself gradually changed so that it remained facing the camera throughout the video. This is a very obvious visual error.

In my view, this failure may be closely related to the generation method described in the paper. Unlike the previous failure cases, which may have resulted from insufficient training data or an incorrect understanding of the prompt, this error appears to be connected to the model’s method of constructing temporal motion. The method first generates images or key frames and then progressively inserts intermediate frames. As a result, the model can produce locally plausible images while failing to preserve the correct three-dimensional structure and viewpoint changes throughout the complete camera movement.

Another possible reason why the model could not generate two complete rotations is that its video generation is essentially based on sequences of generated images. If the statue remains facing the camera throughout the video, the model has no clear visual reference for determining how far the camera has rotated. A 360-degree rotation and a 720-degree rotation may therefore appear very similar in the individual frames. Without maintaining a consistent three-dimensional representation of the statue and its surroundings, the model cannot reliably determine whether the camera has completed one rotation or two.

4, bias
This experiment was relatively simple. I asked the model to generate a handsome man.

I used the same prompt for each test and changed only the random seed. My expectation was that the model might repeatedly generate handsome men with the same ethnicity, skin tone, and similar facial characteristics.

The results were consistent with this expectation. I kept five generated videos, and the men in all five videos had almost identical characteristics:

- white.
- slightly long short hair with similar hairstyles.
- beard in almost the same areas.

Although five samples are not sufficient to draw a broad statistical conclusion, the results show a clear tendency in how the model interprets the word **“handsome.”** Instead of producing people with diverse appearances, ethnicities, and skin tones, it repeatedly generated a narrow and highly similar type of man. This suggests that the model contains a noticeable representational and beauty-standard bias.







### Proposed Improvement: ChatGPT-Assisted Input Processing

Because the failure cases demonstrated very basic limitations in the model, I was not confident that I could directly improve the model itself. Retraining or modifying the internal structure of the model would also require computational resources beyond the scope of this project. Therefore, my proposed improvement focuses on the input process.

The basic idea is to connect the input interface to ChatGPT. ChatGPT first processes the user’s request according to the characteristics and limitations of CogVideo. The improved prompt and generation parameters are then sent to the video-generation model.

This input-side improvement contains three main components.

#### 1. Simplified parameter input

Under the current command-line interface, the user must manually provide many generation parameters. In the improved interface, the user should be able to enter only the text prompt. The system would automatically use suitable default values for parameters such as the inference steps, guidance scale, frame rate, and data type.

If the user wants to change a particular parameter, they can specify only that parameter. All unspecified parameters will continue to use their default values. This would simplify the input process and reduce unnecessary configuration.

#### 2. Dynamic selection of video length

The frame count is also an important parameter. As demonstrated by the camera-rotation experiment, if the generated video is too short, the model may not have enough time to complete the requested movement.

Therefore, ChatGPT could analyze the complexity and expected duration of the requested action before generation. A simple action could use a shorter video, while a longer or more complicated sequence, such as a 720-degree camera rotation, would be assigned more frames.

The frame count should not simply be increased without limitation, because the 161-frame experiment showed that an excessively long generation could seriously reduce visual quality. Instead, the system should select the video length dynamically according to the requested action and the practical limitations of the model.

#### 3. Temporal decomposition of actions

As discussed in the paper, even a seemingly simple action may consist of several temporally ordered stages. A short prompt may not clearly explain this internal sequence to the model.

ChatGPT could therefore divide a complex action into several ordered steps and rewrite the original prompt accordingly. For example, the instruction “a car overtakes another car and makes a U-turn” could be expanded into the following sequence:

1. The first car approaches the second car from behind.
2. It moves into the neighboring lane.
3. It overtakes the second car.
4. It continues forward for a short distance.
5. It gradually makes a U-turn.
6. It drives back toward the second car.

This preprocessing step would make the temporal order more explicit and may help the video model avoid missing actions or generating them in the wrong order.

Overall, the improved pipeline would be:

**User input → ChatGPT analyzes the request → parameters and video length are selected → complex actions are divided into ordered stages → the improved prompt is sent to CogVideo → video generation**

This approach does not change or retrain the original model. Instead, it attempts to improve generation by providing clearer prompts and more appropriate generation settings.

改善 Bias
给模型加入 Negative Prompt
v2v改善模型
chatgpt prompt工程

修改v2v参数 
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            # num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)

profile_index = (seed * 17 + 11) % 30

SEED=52

IMPROVED_PROMPT=$(python prompt_optimizer.py \
  --mode bias \
  --seed "$SEED" \
  --prompt "A head-and-shoulders portrait video of a handsome adult man looking at the camera in a well-lit studio, with subtle natural movement.")

echo "$IMPROVED_PROMPT"

python inference/cli_demo.py \
  --prompt "$IMPROVED_PROMPT" \
  --model_path THUDM/CogVideoX-5b \
  --generate_type t2v \
  --num_frames 49 \
  --num_inference_steps 50 \
  --guidance_scale 6.0 \
  --fps 8 \
  --dtype float16 \
  --seed "$SEED" \
  --output_path "./tests/improve/bias/test6.mp4"