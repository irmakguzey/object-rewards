# Bridging the Human-to-Robot Dexterity Gap through Object-Oriented Rewards
Official implementation of the project **HuDOR: Bridging the Human-to-Robot Dexterity Gap through Object-Oriented Rewards**.

* Website: [https://object-rewards.github.io](https://object-rewards.github.io)
* arXiv: [https://arxiv.org/abs/2410.23289](https://arxiv.org/abs/2410.23289)

[ ] Add some of the results here

## Installation
- [ ] Provide instructions on how to install submodules.
- [ ] Briefly mention how to install the Oculus app (or link to OpenTeach instructions). Consider requiring users to submit a form for access—this might be more practical.

## Data Collection
To collect human demonstrations, after downloading and installing the HumanBot app, you can run:

```bash
python submodules/Open-Teach-HuDOR/data_collect.py storage_path=<desired-storage-path> demo_num=<demo-number>
```

Further adjustments to data collection parameters can be made in the Hydra config file located at: `submodules/Open-Teach-HuDOR/configs/collect_data.yaml`.

During data collection, follow these steps:
* Focus on the ArUco marker on the operation table for approximately 5-6 seconds before starting object manipulation. This helps establish the transformation between the VR headset and the world frame.
* Pinch your right index and thumb fingers to indicate the start of the demonstration.
* Proceed to manipulate the object.
* Pinch your right index and thumb fingers again to signal the end of the demonstration.

