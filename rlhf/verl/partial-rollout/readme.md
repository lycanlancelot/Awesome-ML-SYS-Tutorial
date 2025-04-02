# Kimi K1.5: Long Context RL 的成功实践

感谢 kimi 团队的这篇雄文。大概在 DeepSeek 开源 R1 的同一时间，就有许多朋友向我推荐过 K1.5 的技术报告。苦于工作繁琐，一直没有空拜读。正好最近去西雅图出差，在 LAX 往返 SEA 的路上终于有时间虔诚地读完这一工作。越读起来，越有一种文人相赏，难逢知音的感觉。

本科毕业以来，确实有感受到自己在团队协作和个人能力上的长足长进。但是博士入学之后，我一直没有什么高强度投入的工作发表，未免感到焦虑。今天读完这样的雄文，心情大爽。希望自己能在余下的科研生涯中多参与这样具有开源精神的重磅工作。能让自己的名字出现在此番工作的作者名录之上，不比多发几篇 XXX 或者 XXXX 的论文强？当然，这就又带来了在大项目中，如何证明自己 credit 的问题。不过，我总归相信自己的想法仍是大有裨益的。

絮絮叨叨说了这么多，这篇文章主要复盘自己拜读 K1.5 技术报告的思索。由于是技术报告，这篇扎实的文章涵盖了从数据、训练方法到训练系统的方方面面，读完真是余音绕梁，不绝如缕。

## RL Recipe

K1.5 的训练可以细分为 pretrain，vanila SFT，long-CoT SFT 和 RL 四个阶段。技术报告主要讲述的是 RL 阶段的故事。

### RL prompt 选择

高质量的 RL prompt 需要 diver, balance and accurate to evaluate。为了决定每个 prompt 的难度，作者采用一个 SFT 模型在较高的 temperature 下生成 10 次答案，以 10 次内的通过率作为难度。此外，一些复杂的推理问题可能通过错误的推理过程也能得出正确答案。为了避免此类 reward hacking，作者进一步确保每个 prompt 的 reasoning path 和 final answer 都能被准确验证。作者先排除了容易出现此类错误的题目，例如多选题、判断题和证明题。然后，作者进一步过滤掉一些容易猜测出答案的问题。具体来说，给模型 8 次机会，如果在没有 CoT 的情况下，有超过 1 次可以直接给出答案，就将其移除。

### Long-CoT SFT

作者通过 prompt engineering 构建了一个 multi-modal long-CoT warmup dataset，来让模型初步学会这几种推理能力，evaluation，reflection，exploration 和 planning。

### Length Penalty

在进行 long context RL 训练的过程中，如果不对模型的输出长度做出控制，会很容易观测到 answer length 的显著增加。虽然这带来了更好的性能，但过长的推理过程在训练和推理时成本高昂，而且人类通常不倾向于过度思考。因此，作者引入了一个长度惩罚项，来控制模型的输出长度：

$$ \text{len\_reward}(i) = \begin{cases} \lambda & \text{if } r(x, y_i, y^*) = 1 \\ \min(0, \lambda) & \text{if } r(x, y_i, y^*) = 0 \end{cases} $$

$$ \lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}} $$

分开想想这几种情况：

1. 最长的推理过程，正确的答案。$len\_reward = -0.5$，抑制模型生成此推理过程。
2. 最长的推理过程，错误的答案。$len\_reward = -0.5$，抑制模型生成此推理过程。
3. 最短的推理过程，正确的答案。$len\_reward = 0.5$，鼓励模型生成此推理过程。
4. 最短的推理过程，错误的答案。$len\_reward = 0$，对模型没有影响。

对于长度在 $\frac{\text{max\_len} + \text{min\_len}}{2}$ 以下的的推理过程，如果答案错误，则 length reward 为 0，如果答案正确，则 length reward 大于 0 且随着 length 递减。超过这个长度的 reasoning path，无论答案正误与否，都给一样的负分 length reward。

### 采样策略

虽然 RL 本身具有较好的采样特性，也即难度更高的问题会提供更大的梯度，但其训练效率仍然有限。一些明确的先验采样方法可能会带来更大的性能提升。作者采用的方案有：

1. 课程学习：从较简单的任务开始训练，逐步过渡到更具挑战性的任务。由于初始 RL 模型的性能有限，在非常困难的问题上花费有限的计算预算往往只能产生很少的正确样本，从而导致较低的训练效率。同时，用于训练的数据天然就包含难度标签，进一步让难度递增学习策略直观且有效。
2. 优先采样：成功率较低的问题采样更多次。我们跟踪每个问题 $i$ 的成功率 $s_i$，并按照 $1 - s_i$ 的比例采样问题，从而使成功率较低的问题获得更高的采样概率。这将模型的努力集中在它最薄弱的领域，从而加速学习并提升整体性能。

### Code, Math and Visual Data

由于爬虫的合规性限制，很多网上得到的 coding 问题没有测例。为此，作者使用 CYaRon1 来生成测例，作为 reward。这当然，需要很多严苛的假设，譬如不需要特殊的评判，并且这些问题有可用的 ground truth solution，以便利用这些 solution 生成更高质量的测试用例。

至于数学，评估数学问题的一大挑战是，不同的表达形式可能代表相同的答案。例如，$a^2 - 4$ 和 $(a + 2)(a - 2)$ 可能都是同一个问题的有效解。为了提高奖励模型的评分准确性，作者尝试了两种方法：

1. Classic RM：参考 InstructGPT，作者实现了一个基于 value-head 的奖励模型，并收集了约 80 万个数据点进行微调。该模型最终以 question, answer, reference answer 作为输入，输出一个标量，指示响应是否正确。
2. Chain-of-Thought RM：最近的一些工作证明，加入了 CoT 的 reward model 在需要细微正确性标准的任务上显著更优。作者继续收集了约 80 万个带有 CoT 的数据集来微调 reward model。在与 Classic RM 相同输入的基础上，CoT model 明确生成逐步推理过程，最后以 JSON 格式提供最终的 reward。在作者的手动抽查中，Classic RM 的准确率约为 84.4，而 CoT RM 达到了 98.5 的准确率。最终，他们采用了 CoT RM 以确保更正确的反馈。

视觉强化学习 Vision RL 的数据主要来源于三大类：Real-world data，Synthetic visual reasoning data 以及 Text-rendered data。其中，text-rendered data 颇有意思：将文本内容转换为图片，专门强化模型基于跨模态的文本推理能力。通过将文本文档、代码片段和结构化数据转换为图像，作者格外训练了模型处理文本密集图像的能力。