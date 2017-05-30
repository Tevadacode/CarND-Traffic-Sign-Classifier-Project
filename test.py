# read training, validation, test data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
file = "thirty1.JPG"
img = mpimg.imread(file)
plt.plot([1,2,3])
plt.subplot(211)
plt.title("Test Image")
plt.xlabel("30 km/h")
plt.imshow(img)
plt.subplot(212)
plt.plot(range(20))
plt.show()
