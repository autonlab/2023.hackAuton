# 2023.hackAuton
This repository hosts all the hacks from the 2023 hackAuton event!

## How to Submit

Submitting your hack is easy, follow these steps:

1. One person from your team should sign up for a Github account if you don't already have one. You will need it to submit your code.
2. Visit the [hackAuton repository on Github](https://github.com/autonlab/2023.hackAuton) and, on the top right, click "Fork" to make a new copy of the repository to your account.
3. Checkout your repository to a computer that has your code on it. The command will be something like:
    ```
    git clone https://github.com/your_username/2023.hackAuton
    ```
    You need to check out your fork because you cannot commit directly to the Auton Lab repository.
4. Create a new directory in the repository you just checked out with your team name, and add your code to that directory. If you used a Jupyter notebook it is very helpful *not* to clear your results so they can serve as a baseline for someone trying to reproduce them.
5. Please also put your final report in with your code. Please include the source and a PDF. The source will allow us to more easily aggregate the papers later.
6. Figure out what license you want to give to your code. If you used GPL'd libraries you might have to choose a GPL license. Creative Commons has a [nice website to help you choose a license](https://choosealicense.com/) or you can choose a Creative Commons license [using their wizard](https://creativecommons.org/share-your-work/) to set exactly the permissions you want. Copy the license text into your team directory in a file named something like `License.txt`. *Remember to put in the year and the names of your team members in the license so you're properly credited!* Now commit those changes and push it to Github!
    ```
    git commit -am "Added code and license for team Your Team Name"
    git push
    ```
7. You're almost done! Your code is in your Github fork but not in Auton Lab's repository. Visit your fork on Github (https://github.com/your_username/2023.hackAuton). On the top left, just above the file list, there is a button labeled "New pull request". Click that button and you should see all of your additions and an indication that the pull request is going to our repository. Once you create the pull request you're done! We will review your submission and add it to the official repository.
