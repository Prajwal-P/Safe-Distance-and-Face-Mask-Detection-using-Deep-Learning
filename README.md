# Safe-Distance-and-Face-Mask-Detection-using-Deep-Learning
To implement a system that provides quick and efficient results on whether people are maintaining social distance and to check if people are wearing face masks properly or not.

---

## Steps for contributing:
1. Assign a issue to yourself.
2. Create a new branch from ```main``` branch. Follow a branch name convention like ```feature/[BRANCH_NAME]```, here have a ```BRANCH_NAME``` that's short and relates to the issue you are solving.
    - Example: Branch name for solving [#1] could be ```feature/load-dataset```.
3. Make changes in your branch, commit and push them.
4. Create a Pull Request from your branch to main branch. If any two members of our team approves your changes you are free to merge the changes with main branch.

NOTE: If there are any comments in your pull request, please resolve them before merging. Also consider changing notification setting for this repo to watch all changes to follow all changes done.

---

## Things we have to enhace in this model or project
- [ ] Improve frame rate of output.
- [ ] Improve output from mask detector model. The that we are getting now is fluctuating between ```mask``` and ```without-mask```, we should have a model that gives a stable output.
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

## References
### Other projects
- https://github.com/Prajwal-P/Social-Distancing-and-Face-Mask-Detection
- https://github.com/rohanrao619/Social_Distancing_with_AI
- https://github.com/adityap27/face-mask-detector
- https://github.com/smahesh29/Gender-and-Age-Detection - Try out face detector model from this project

### Dataset
https://www.kaggle.com/shantanu1118/face-mask-detection-dataset-with-4k-samples


[#1]: https://github.com/Prajwal-P/Safe-Distance-and-Face-Mask-Detection-using-Deep-Learning/issues/1
