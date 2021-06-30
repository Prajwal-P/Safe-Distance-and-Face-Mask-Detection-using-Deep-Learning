# Safe-Distance-and-Face-Mask-Detection-using-Deep-Learning
To implement a system that provides quick and efficient results on whether people are maintaining social distance and to check if people are wearing face masks properly or not.

---
## Steps for contributing:
1. Assign a issue to yourself.
2. Create a new branch from `main` branch. Follow a branch name convention like `feature/[BRANCH_NAME]`, here have a `BRANCH_NAME` that's short and relates to the issue you are solving.
    - Example: Branch name for solving [#1] could be `feature/load-dataset`.
3. Make changes in your branch, commit and push them.
4. Create a Pull Request from your branch to main branch. If any two members of our team approves your changes you are free to merge the changes with main branch.

### NOTE:
- Requirements file should be manually filled, with only package names and without it's version so that latest version will be inatalled.
- If there are any comments in your pull request, please resolve them before merging. Also consider changing notification setting for this repo to watch all changes to follow all changes done.
- Please follow these steps while creating a [pull request]:
    - After clicking new pull request check the source and destination branches.
    - Give a meaninful title to the PR.
    - Add a small discription to the PR linking the [issue]. Refer [this] for more info about why to link.
    - On the right sidebar add the reviewers you want your PR to be reviewed from and also assign the PR to yourself.
- While merging the PR use the option to squash your commits and merge and also to delete your feature branch. Here your PR title will be used as commit message in main so make sure you have given a proper PR title.
- In commit message instead of giving `completed` or `resolved comments` give a meaning full commit message like `added load dataset` or `updated dataset file path`.

---
## Things we have to enhace in this model or project
- [ ] Improve frame rate of output.
- [ ] Improve output from mask detector model. The that we are getting now is fluctuating between `mask` and `without-mask`, we should have a model that gives a stable output.
- [ ] Find social distancing without giving a hard coded pixel distance between the centroides of detected people.
- [ ] Should have three classification for mask detection:
    - Properly worn / With mask
    - Improperly worn
    - Without mask

---
## Stuff to be kept in mind while working
- To use some real life dataset taken by us.
- Why are we changing to 1 nuron in output layer from 2?
- Change threshold confidence score to 60%.

---
### Important commands
1. `conda create --name <env> -c conda-forge --file <this file>`
    > To create conda environment and install packages.

---
## References
### Other projects
-   https://github.com/Prajwal-P/Social-Distancing-and-Face-Mask-Detection
-   https://github.com/rohanrao619/Social_Distancing_with_AI
-   https://github.com/adityap27/face-mask-detector
-   https://github.com/smahesh29/Gender-and-Age-Detection - Try out face detector model from this project
### Dataset
https://www.kaggle.com/shantanu1118/face-mask-detection-dataset-with-4k-samples



[#1]: https://github.com/Prajwal-P/Safe-Distance-and-Face-Mask-Detection-using-Deep-Learning/issues/1
[pull request]: https://github.com/Prajwal-P/Safe-Distance-and-Face-Mask-Detection-using-Deep-Learning/pulls
[issue]: https://github.com/Prajwal-P/Safe-Distance-and-Face-Mask-Detection-using-Deep-Learning/issues
[this]: https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
