given a wav file, create a python program to perform stft on the wav, keeping the phase. and then map the magnitude to mel, and then run a convex optimization on the mel to extract the ridge: 1. 从沿频率轴做局部峰值检测 得到垂直峰值掩码  2. 硬约束会强制脊线只能出现在峰值候选位置上 3. 稀疏 + 水平 TV。keep k(default 5) ridges per frame. Then followed by a phase where gaussian ball are planted at equal space along time axis, and then optimized to minimize mel l2 loss with gt. gaussian ball are encouraged to grow vertically and not get too wide.
然后，map from ridge and gaussian on mel back to stft magnitude, and then 利用 stft GT phase ，恢复出 wav.
最后生成 html side-by-side 比较两个 wav 的 mel 谱，和在网页能播放 wav。

---
将以上变成 gemini-cli/codex 能用的 SPEC markdown。
