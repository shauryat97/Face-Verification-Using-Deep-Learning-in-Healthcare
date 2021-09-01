# Face-Verification-Using-Deep-Learning-in-Healthcare

This project was done as a part of a Hackathon for **Deep Learning for Computer Vision** course at IISc.
My team was among the **top three** teams out of  all the  participating teams in class.

# Introduction

An NGO funds cataract surgeries for the underprivileged. They fund eye hospitals on basis of free surgeries performed. Funds are released, based on the number of operations performed.

The NGO needs to ensure that the funds are released only for genuine surgeries. As proof that the surgery is actually performed, the NGO currently requests eye hospitals to send in pre-operation (pre-op) and post-operation (post-op) photographs of the patient’s face for each surgery.

The NGO then verifies the pre-op and post-op images to ensure that the surgery has actually been performed before releasing funds. They also need to verify that photographs of same patients are not being sent again and again to claim funds fraudulently.

They need a system that will help in comparing these photographs to highlight cases with a high probability of fraud. These can then be further investigated manually.

# Problem statement

Pre-op and post-op photographs will be provided. (See figure below) The NGO needs a system that will help compare the pre and post-operative images and identify that this is the same patient with a high level of confidence.

The post-operative pictures captured, have a part of the patients' face covered with a patch. The existing facial recognition algorithms available require all the parts of the face to be visible. Hence when a part of the face is covered with the patch, these algorithms are not very effective.

The problem is to find a solution that will be able to detect and match the patients from pre-op and post-op images to confirm it is the same patient (even when a part of the face is covered with a patch )

![eye_patient](https://user-images.githubusercontent.com/73641247/131727255-5899a758-8273-4cc5-b7f9-0d1cec8ab99b.jpg)
![Sample Image](https://github.com/shauryat97/Face-Verification-Using-Deep-Learning-in-Healthcare/blob/main/eye_patient.jpg)

