# Human Detection (using HOG)
## What is HOG?
HOG는 ***Histogram of Oriented Gradients***의 약자로 한국어로 직역을 하면 ***기울임 강조 히스토그램(?)*** 이라고 할 수 있을거 같다. 사실 명확히 말하면 HOG **Feature Descriptor** 라는 말이 붙는게 맞다. 이에 대한 이유는 사실 2차시 수업 자료에 작성을 해두기는 했다. Feature Descriptor 특징 설명자(?) 의 종류는 HOG 외에도 굉장히 다양하면 특징을 잡는 하나의 기술이라는 것 중에 하나이기 때문이다. 뭐 아무튼 작동 원리를 설명해주면서 이 코드에 대한 설명을 하겠다.

# My Code
## plot_hog.py
해당 파일에 있는 코드의 역할은 HOG Feature Descriptor를 통해 사람을 인식하기 이전에 먼저 사람의 모습을 전지적 CV 관점에서 보기 위한 코드이다. 쉽게 말해 이해를 돕기 위한 코드라고 하면 되게 쉽다.

코드 설명은 부분부분 나눠가면서 설명 하겠다.
```python
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

image = plt.imread('tyler.jpg')

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
```
일단 전체 코드는 다음과 같고 천천히 부분부분 자세히 설명하 예정
```python
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

image = plt.imread('tyler.png')
```
일단 HOG를 통해 사람을 인식하기 전의 모습을 보기 위해서는 cv가 아닌 skimage를 다운로드 받아줘야 한다. 이는 pip를 통해 설치가 가능하니 찾아보면 나온다. plt는 필수는 아니지만 결과 화면을 보기 좋게 보여주기 때문에 선택 사항이다. cv를 써도 되지만 코드 또한 변경을 해줘야 한다. 뭐 그건 개인 취향대로..

plt는 통해 tyler.png를 갖고 온다.
```python
d, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
```
d, hog_image가 이 코드의 주 목적과 결과물을 담당한다고 할 수 있을거 같다. skimage로 부터 hog라는 함수를 갖고온다.

자 이제부터 함수의 인자값을 하나하나 해석(?)을 하겠다. 사진을 읽어오고, 방향성을 몇으로 할 것인지 정한다. pixels_per_cell은 한 cell당 몇 픽셀로 해줄지 정해준다. 이 부분은 수업자료에 들어가면 그림으로 설명을 해두었다. 그래도 모르겠다면 번역기를 사용해서라도 <a href="https://learnopencv.com/histogram-of-oriented-gradients/">이 문서</a> 를 꼭 읽어보길 바란다. HOG에 대한 내용을 굉장히 잘 담아둔 문서이다.

cells_per_block은 각 cells의 거리(?)를 몇으로 해줄지 정해준다. ~~위에 말한 문서를 읽어보면 이해가 될 것이다.~~ visualise는 시각화를 해줄 것인가 안할 것인가를 물어보는 건데 우리는 결과 값을 봐야 하기 때문에 필요하다. multichannel는 색을 말하는데 크게 상관은 없다.

fig와 (ax1, ax2)는 시각화를 시키기 위한 부분이다. plot을 통해 창을 띄어주기 전에 준비하는 과정 중 하나이다. 크게 중요한 것은 없기 때문에 패스.

```python
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
```
ax1에 해당 되는 부분은 패스한다. 그냥 화면을 띄어주는 것 외에는 크게 없기 때문이다. hog_image_rescaled는 위에서 HOG Descripitor를 한번 거친 이미지의 크기를 한번 조정을 해준다.
```python
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
```
최종적으로 변환이 된 사진을 보여준다. 

## detector.py
본격적으로 사람을 인식하는 코드이다. 원리 또한 코드 설명을 하면서 차근 차근 작성 하도록 하겠다. 일단 전체 코드는 아래와 같다.
```python
import cv2
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread('tyler.png')

image = imutils.resize(image, width=min(400, image.shape[1]))
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), 
        scale=1.05)

for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
부분부분 나눠가면서 설명을 하도록 하겠다.

```python
import cv2
import imutils
```
필요한 모듈은 사실 1개 이다. cv2 모듈만 사용을 해도 작동 되는 코드이다. 그런데 imtuils를 사용하는 이유는 화면 비율 모듈이 굉장히 편하기 때문이다. 이는 아래에서 다시 설명 하도록 하겠다.

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread('tyler.png')
```
일단 cv2 모듈에서 HOGDescriptor 함수를 갖고 와준다. 그러고 갖고온 함수에서 setSVMDetector를 통해 어떤 객체를 인식을 할지 정해준다. 우리가 인식할 객체는 현재 사람이기 때문에 *cv2.HOGDescriptor_getDefaultPeopleDetector()* 를 통해 객체를 지정해준다.

여기서 혼동이 되기 전에 미리 알아야 할 점은 **HOG는 객체인식 설명 기술 중 하나 일 뿐 절대 사람인식 기술이 아니다.** 무슨 말이냐면 HOG를 통해서 다른 객체 또한 인식이 가능하다는 것이다. 혼동이 되면 절대로 안된다. HOG를 통해 자동차를 인식하는 경우도 있고 동물을 인식하는 경우도 있기 때문에 반드시 혼동이 되서는 안된다.

image는 tyler.png 파일을 읽어와준다.

```python
image = imutils.resize(image, width=min(400, image.shape[1]))
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
```
여기서 내가 **굳이** imtuils 모듈을 사용하는 이유가 있다. image와 세로 혹은 가로의 크기만 맞게 지정을 해주면 그 비율에 맞게 사진의 크기를 조정해준다. width에 min(400, image.shape[1]) 으로 한 이유는 사진의 비율이 항상 동일하진 않기 때문이다. 그렇기 때문에 최소값에 맞게 조정을 해주는 것 이다.

(regions, _) 는 위에서 지정해준 객체 인식기를 통해 나올 값이다. 이 부분이 가장 중요한데 hog.detectMultiScale 함수의 각 파라미터를 설명을 하면서 짚고 넘어가겠다. 가장 중요한 image가 있고, winStride가 있다. winStride는 HOG Descripitor가 객체를 인식을 할 때 아래와 같은 방식으로 인식을 한다.
<div align="center">

![winStride_Example](https://pyimagesearch.com/wp-content/uploads/2014/10/sliding_window_example.gif)

*Gif image from __pyimagesearch__*
</div>

위와 같은 방식으로 인식을 할 때 이에 크기를 몇으로 해줄 것인지를 정해주는 것이다. 즉, winStride의 크기가 크면 속도는 빨라지지만 인식율를 떨어지고 크키가 작으면 속도는 느려지지만 인식율은 높아진다고 할 수 있다. 이것 또한 확답이 어려운 것이 어떤 사진을 하냐에 따라 다른데 이 부분은 뒤에서 다루기로.

padding은 필수는 아니지만 적당히 padding을 추가해주면 객체 인식율을 어느정도 올릴수 있다고 한다. **<a href="https://hal.inria.fr/inria-00548512/document">Dalal and Triggs in their 2005 CVPR paper</a>**

일반적으로는 (8, 8), (16,16), (24, 24), (32, 32)의 패딩을 사용한다. 이는 각자 직접 실행을 해보면서 적당한 값을 찾는 것이 좋다.

scale 파라미터도 필수는 아니다. 하지만 나름 중요한 내용이기에 짚고는 넘어가야 한다. scale 개념을 가장 잘 표현한 사진은 이것이다. <br>
<div align="center">

![scale parameter example](https://www.pyimagesearch.com/wp-content/uploads/2015/03/pyramid_example.png)

*Image from __pyimagesearch__*

</div>

scale을 몇으로 하냐에 따라 각 픽셀을 읽는 개수가 달라진다. 즉 hog가 읽는 숫자의 개수 혹은 크기들이 다 달라질 수 있다는 것 이다. scale를 크게 하면 레이어의 갯수가 증가하며 계산을 해야할 양 또한 증가한다. 반대로 scale를 작게 하면 레이어의 갯수가 줄어들며 계산을 해야할 양도 줄어든다. 

사실 이 외에도 다다들 파라미터들이 있다. 이 코드에서는 쓰이질 않아 패스한다.

```python
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
regions의 좌표들을 통해서 총 4개의 점을 얻어낸다. 각 좌표가 필요한 이유는 객체가 인식이 된 부분에 표시를 해주기 위해서이다. 그래서 cv2.rectangle를 통해서 객체의 위치를 보여준다.

나머지 부분은 최종적인 화면을 띄어주는 부분이기에 패스.

# My Comments
HOG Descriptor를 이제는 활용하는 케이스는 거의 못 봤다. 대신에 왜 이 기술을 알아야 하는지 묻는다면 HOG가 이제는 ***LEGACY*** 라고 불릴 정도로 기술의 방법과 작동 원리 만큼은 확실히 좋다. 

또한 앞으로 나오고 있는 Feature Descrpitor 들 또한 HOG를 조상(?)처럼 여기는 경우가 종종 있기 때문이다. 그러니 이런 기술의 작동 원리를 알아두면 다른 기술을 공부하는데에 이해하기가 쉽다.

추가적으로 OpenCV에 대한 공부를 좀 더 심도 있게 하고 싶다면 개인적으로는 영문을 직접 해석해가면서 읽는걸 추천한다. 물론 한국어로 된 좋은 문서들도 많다. 하지만 아직 OpenCV 알고리즘의 작동 원리나 기술에 대한 내용은 아직 한국어보다는 영어가 많은건 사실이다. 그로 인해 나도 도움이 될지도 모르겠지만 이런식으로 계속 이런 글을 작성해가는거다.

나에게는 되게 오랜만에 하는 간단한 내용이지라도 누구에게는 처음보는 코드가 될 수 있기 때문이다. 간단하지만 그 안에는 굉장히 많은 일들이 돌아가는 OpenCV 코드를 보면서 작동 원리를 꼭 알아 갔으면 좋겠다는 작은 나의 소신이다.