# IDT-DDSA

# IMAGE-DERAINING-TRANSFORMER-USING-DYNAMIC-DUAL-SELF-ATTENTION

<hr />

> **Abstract:** *Recently, Transformer-based architecture has been introduced into single image deraining task due to its advantage
in modeling non-local information. However, existing approaches tend to integrate global features based on a dense
self-attention strategy since it tend to uses all similarities of
the tokens between the queries and keys. In fact, this strategy
leads to ignoring the most relevant information and inducing blurry effect by the irrelevant representations during the
feature aggregation. To this end, this paper proposes an effective image deraining Transformer with dynamic dual selfattention (DDSA), which combines both dense and sparse
attention strategies to better facilitate clear image reconstruction. Specifically, we only select the most useful similarity
values based on top-k approximate calculation to achieve
sparse attention. In addition, we also develop a novel spatialenhanced feed-forward network (SEFN) to further obtain a
more accurate representation for achieving high-quality derained results. Extensive experiments on benchmark datasets
demonstrate the effectiveness of our proposed method.* 
<hr />

## Network Architecture
![arch](https://github.com/zt-fan/IDT-DDSA/assets/90734659/a7a9006d-e5ae-4925-8923-290dafde8413)


## Installation
See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## Results
![table](https://github.com/zt-fan/IDT-DDSA/assets/90734659/0fd00425-ef18-4c58-82a9-6163deda86b5)

![pic](https://github.com/zt-fan/IDT-DDSA/assets/90734659/278f448b-49b1-46aa-b0d9-dc51127ced83)
