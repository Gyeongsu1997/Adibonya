## 캡스톤 디자인 프로젝트: 집중도 판단 서비스, '어디보냐'
온라인 강의를 통한 학습 시에 웹캠을 통해 집중도를 측정하여 그 결과를 점수로 보여주고 기록하는 서비스입니다.

[상세 설명](https://gyeongsu1997.github.io/django/adibonya/)

## 사용방법
- 터미널에서 명령어 입력

  1.git clone https://github.com/Gyeongsu1997/Adibonya.git [폴더명]

  2.cd [폴더명]

  3.pip install -r requirements.txt

  4.python manage.py migrate
  
  5.python manage.py runserver

- 브라우저에서 http://127.0.0.1:8000/ 로 접속하여 서비스 이용

## 개발 배경
인천대학교 컴퓨터공학부 졸업작품으로 캡스톤 디자인 프로젝트를 수행하였다. 우리 InuIT 팀은 온라인 강의를 통한 학습 시에 웹캠을 통해 집중도를 측정하여 그 결과를 점수로 보여주고 기록하는 서비스를 만들기로 하였다. 집중도 판단을 위해 딥러닝 모델을 사용하였다. 딥러닝 모델은 집중하는 모습을 담은 사진과 집중하지 않는 모습을 담은 사진을 각각 0과 1로 라벨링한 훈련 데이터셋으로 ResNet-50 신경망에 대한 전이학습을 통해 제작되었다. 이는 아래 링크의 코드를 참고하였다.
- [[python] 쉽고 간단하게 마스크 착용 유무 판별기 만들기](https://bskyvision.com/1082)

이 페이지에서는 서비스를 위한 웹 개발 과정에 중점을 두어 기록하였다. 작품개요는 아래 포스터와 같다. 
<img src="../../images/2022-06-22-capstone/poster.PNG" alt="poster"  />

## 사용 기술
Python 기반 웹 프레임워크인 Django를 사용하였고 mashup-template.com에서 제공하는 Mountain HTML 템플릿을 사용하였다. DBMS는 Django에서 기본으로 제공하는 SQLite를 사용하였다.
- [Mashup Template](http://www.mashup-template.com/)

## 주요 기능별 사용 시나리오
- 메인 페이지
  ![index](../../images/2022-06-22-capstone/index.PNG)
  메인 페이지에서는 서비스의 개요와 주요 기능들을 소개한다. 중앙 또는 우측 상단의 로그인 버튼을 통해 로그인 페이지로 이동할 수 있다. 로그인을 하지 않은 사용자는 서비스를 이용할 수   없도록 집중도 측정 버튼을 눌러도 로그인 페이지로 이동한다.

- 로그인 및 회원가입
  <img src="../../images/2022-06-22-capstone/login.PNG" alt="login"  />
  ![signup](../../images/2022-06-22-capstone/signup.PNG)

- 마이 페이지
  ![mypage](../../images/2022-06-22-capstone/mypage.PNG)
  로그인한 사용자는 마이 페이지로 이동한다. 마이 페이지에서는 과거 학습기록을 캘린더에 점수에 따른 이모티콘 형태로 표시해 학습 이력을 조회 및 관리할 수 있다. 캘린더 제작은 huiwenteo.com의 게시글을 참고하였다.
  - [HOW TO CREATE A CALENDAR USING DJANGO](https://www.huiwenteo.com/normal/2018/07/24/django-calendar.html)

- 집중도 측정
  ![project](../../images/2022-06-22-capstone/project.PNG)
  집중도 측정 페이지로 들어가면 시작 버튼을 눌러 측정을 시작할 수 있다. 측정 도중 다른 페이지로 이동하게 되면 기록된 점수가 제대로 처리되지 않기 때문에 상단에 내비게이션 바를 모두 제거하였다.
    - 집중 판단 모습
      ![concentrate](../../images/2022-06-22-capstone/concentrate.PNG)
    - 비집중 판단 모습
      ![noconcentrate](../../images/2022-06-22-capstone/noconcentrate.PNG)
   ![stop](../../images/2022-06-22-capstone/stop.PNG)
   일시정지 버튼을 누르고 다시 정지 버튼을 누르면 나오는 알림 메시지에서 확인 버튼을 누르면 결과 페이지로 이동한다.
  
- 결과 페이지
  ![result](../../images/2022-06-22-capstone/result.PNG)
  결과 페이지에서는 집중도 측정 결과를 분 단위로 점수화해 그린 차트가 보여진다.
  ![result_error](../../images/2022-06-22-capstone/result_error.PNG)
  만약 집중도 측정 페이지에서 측정을 하지 않고 바로 정지 버튼을 누른다면 측정 결과가 존재하지 않는다는 메시지가 나타난다.
  
## 설계 및 구현
- 웹캠 스트리밍

  집중도 측정 페이지의 웹캠 스트리밍 기능은 아래 BIPA SORI님의 유튜브 영상을 참고하여 제작하였다.
  - [장고(Django)로 웹캠 비디오 스트리밍](https://www.youtube.com/watch?v=O87Rwc6yc80&list=WL&index=86)

  다만 이 프로젝트에서는 웹캠을 단순히 스트리밍하는 것에 그치는 것이 아니라 웹캠으로부터 얻은 프레임에서 얼굴 부분을 검출하고 전처리 과정을 거친 후 딥러닝 모델을 통해 집중도를 판단하도록 하여 그 결과를 데이터베이스에 저장하는 동시에 바운딩 박스를 씌운 이미지를 스트리밍해야 했다. 위 영상의 코드와 달라진 것은 쓰레드를 사용하지 않았다는 것이다. 위 영상에서는 threading.Thread() 함수를 통해 update 메서드를 실행한다. update 메서드에서는 무한 반복문을 통해 지속적으로 웹캠으로부터 프레임을 읽는다. 필자가 이와 같이 제작하였더니 일시정지 또는 정지 버튼을 눌러도 백그라운드에서 웹캠이 작동하고 있는 문제가 있었다. 그래서 쓰레드를 사용하지 않고 get_frame 메서드에서 프레임을 읽도록 했다. gen() 함수에서 get_frame 메서드를 반복적으로 실행하니 문제가 없을거라 생각했다. 이를 통해 앞서 언급한 문제를 해결할 수 있었다.
  
  > BIPA SORI님의 코드
  
  ```python
  class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

  def gen(camera):
      while True:
          frame = camera.get_frame()
          yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

  @gzip.gzip_page
  def detectme(request):
      try:
          cam = VideoCamera()
          return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
      except:
          print("에러입니다...")
          pass
  ```
  
  > 필자가 작성한 코드
  
  ```python
  class VideoCamera(object):
    def __init__(self):
        self.model = load_model(os.path.join(os.path.dirname(__file__),"..") + '/static/model.h5')
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()

    def get_frame(self, user):
        (self.status, self.frame) = self.video.read()
        
        if (not self.status):
            self.video = cv2.VideoCapture(0)
            return False
        image = self.frame
        face, confidence = cv.detect_face(image)
        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            if 0 <= startX <= image.shape[1] and 0 <= endX <= image.shape[1] and 0 <= startY <= image.shape[0] and 0 <= endY <= image.shape[0]:
                    face_region = image[startY:endY, startX:endX]
                    face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)
                    x = img_to_array(face_region1)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    prediction = self.model.predict(x)
                    
                    if prediction < 0.5:
                        score = TempScore(user=user, score=100)
                        score.save()
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        text = "Concentrate ({:.2f}%)".format((1 - prediction[0][0])*100)
                        cv2.putText(image, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    else:
                        score = TempScore(user=user, score=0)
                        score.save()
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        text = "No concentrate ({:.2f}%)".format(prediction[0][0]*100)
                        cv2.putText(image, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

  def gen(camera, user):
      while True:
          frame = camera.get_frame(user)
          if (not frame):
            continue
          yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

  @gzip.gzip_page        
  def detect(request):
      try:
          cam = VideoCamera()
          return StreamingHttpResponse(gen(cam, request.user), content_type="multipart/x-mixed-replace;boundary=frame")
      except:
          print("error")
          pass
  ```
  
- 데이터베이스

  데이터베이스에는 TempScore와 History라는 2개의 테이블이 존재한다. TempScore는 Temporary Score의 약자로 말 그대로 임시 저장소이다. VideoCamera 클래스의 get_frame 메서드에서는 모델에 의한 예측값이 0.5 미만이면 집중으로 판단하여 사용자, 현재 시간(자동으로 추가)과 함께 100점의 점수를 TempScore에 저장한다. 반대로 모델의 예측값이 0.5 이상이면 비집중으로 판단하여 사용자, 현재 시간과 함께 0점의 점수를 TempScore에 저장한다. 만약 측정이 끝나고 TempScore에 존재하는 기록들을 그대로 놔둔다면 다음 번의 측정이 끝나고 그 결과를 차트로 그릴 때 이전의 기록까지 같이 나타나게 될 것이다. 이러한 문제를 해결하기 위해 점수들을 1분 단위로 평균을 구해 차트를 그리고나면 전체 점수의 평균만을 구하여 History에 저장하고 TempScore의 기록은 모두 삭제되게 하였다. 따라서 TempScore는 임시 저장소가 되는 것이다. 이 과정은 집중도 측정 페이지에서 정지 버튼을 눌러 결과 페이지가 보여질 때 수행된다.
  
- 측정 결과 차트

  결과 페이지에서 보여지는 차트는 원래 팀원 중 다른 한 명이 구현하였다. 초기에는 데이터베이스에 저장된 모든 점수가 차트에 표시되도록 하였는데 데이터베이스에 저장된 점수는 0점 아니면 100점이기 때문에 차트가 마치 사각파와 같은 형태로 불연속적으로 그려졌다. 그래서 필자는 점수들을 분 단위로 나누어 평균을 구하고 이 1분 간격의 평균 점수들만이 차트에 표시되도록 수정하였다. 먼저 TempScore에서 특정 사용자의 데이터를 시간의 오름차순으로 불러와서 첫번째 레코드의 time을 dt라는 변수에 할당한다. 누적 점수를 담을 score라는 변수와 더해진 레코드의 수를 의미하는 count라는 변수를 0으로 초기화한다. for 반복문에서는 레코드를 순차적으로 접근하며 해당 레코드의 분, 시간, 날짜를 dt의 분, 시간, 날짜와 각각 비교한다. 이 값들이 모두 같으면 score에 해당 레코드의 점수를 더하고 count를 1만큼 증가시킨 후 다음 레코드에 접근한다. 만약 이 값들 중 다른 것이 있으면 score를 count로 나누어 그 몫을 avg_score라는 변수에 저장한 후 이를 시간 정보와 함께 score_list에 추가한다. 다시 dt, score 변수에 해당 레코드의 time, score를 대입하고 count를 1로 한 후 다음 레코드에 접근한다. 이때 레코드의 분만 비교하는 것이 아니라 시간까지 비교하는 이유는 다음 생각에 기인한다. 만약 측정 도중 2시 30분에 일시정지 버튼을 누르고 3시 30분에 시작을 눌러 측정을 재개했다면 분만 비교할 때는 3시 30분에 기록된 점수들이 2시 30분의 점수들과 함께 더해져 평균이 구해지는 문제가 발생할 것이다. 날짜를 비교하는 것도 이와 같은 이유에서다. 코드는 아래와 같다.
  
  > result 뷰 함수
  
  ```python
  def result(request):
    stocks = TempScore.objects.filter(user=request.user).order_by('time')

    if len(stocks) == 0:
        return render(request, 'conc/result_error.html')

    score_list = []
    dt = stocks[0].time
    score = 0
    count = 0
    for stock in stocks:
        if dt.minute == stock.time.minute and dt.hour == stock.time.hour and dt.date() == stock.time.date():
            score += stock.score
            count += 1
            continue

        avg_score = score // count
        time_tuple = strptime(str(dt), '%Y-%m-%d %H:%M:%S.%f')
        utc_now = mktime(time_tuple) * 1000
        score_list.append([utc_now, avg_score])

        dt = stock.time
        score = stock.score
        count = 1

    avg_score = score // count
    time_tuple = strptime(str(dt), '%Y-%m-%d %H:%M:%S.%f')
    utc_now = mktime(time_tuple) * 1000
    score_list.append([utc_now, avg_score])

    score_by_minute = np.array(score_list)[:,1]
    total_avg_score = int(score_by_minute.mean())

    timediff = stocks[len(stocks)-1].time - stocks[0].time
    history = History(user=request.user, avg_score=total_avg_score, start_time=stocks[0].time, end_time=stocks[len(stocks)-1].time, duration=timediff.seconds//60)
    history.save() #History에 전체 평균 점수 저장
    stocks.delete() #TempScore에 저장된 기록 삭제

    scoreJson = json.dumps(score_list)
    data = {
        'score': scoreJson,
    }

    return render(request, 'conc/result.html', data)
  ```
  
## 한계
- 집중 여부 판단의 정확도 문제

  본 서비스에서는 사용자의 집중 여부 판단을 전적으로 딥러닝 모델에 의존한다. 모델 제작을 위한 데이터셋으로 사용하기 위해 정면을 바라보는 사진을 집중으로, 정면이 아닌 상,하단 또는 측면을 바라보는 사진을 비집중으로 라벨링을 하였다. 그런데 집중과 비집중은 마스크 착용 여부처럼 뚜렷하게 구분되는 것이 아니므로 모델에 의한 판단도 부정확했다. 예를 들어 고개의 방향이 정면을 향하면서 눈을 감고 있는 경우는 집중하지 않는 것으로 판단해야 하는데 집중하는 것으로 판단하였다. 또한 고개가 정면에서 살짝만 틀어져도 집중하지 않는 것으로 판단하는 등의 문제가 있었다. 이에 더해 마스크를 착용한 경우는 고개의 방향과는 무관하게 대부분 비집중으로 판단하였다. 이러한 문제는 모델을 잘못 만들었기 때문이 아니라 얼굴 이미지만으로 집중 여부를 판단하는 것이 매우 어렵기 때문이라고 생각한다. 기획의 중요성이 여실히 드러나는 부분이다.
  
- 원격 서버가 아닌 로컬 서버 사용

  본 서비스를 이용하려면 사용자가 직접 로컬 서버를 실행하고 루프백 주소의 8000번 포트로 접속해야 한다. 처음 기획은 원격 서버를 사용하여 사용자가 도메인을 통해 접속만 하면 서비스를 이용할 수 있도록 하는 것이었다. AWS를 통해 원격 서버를 구축하고 접속하여 테스트해본 결과 다른 기능은 모두 정상적으로 작동하지만 집중도 측정 페이지의 웹캠 스트리밍 기능만 작동하지 않았다. 이 문제를 해결하기 위해 CORS 설정, SSL 인증서 발급 및 적용 등 여러가지 시도를 해보았지만 결과는 그대로였다. 마침내 찾아낸 원인은 다음과 같다. 본 프로그램에서 웹캠의 제어는 OpenCV 라이브러리의 cv2.VideoCapture() 함수를 통해 이루어진다. 로컬 서버를 구동했을 때 웹캠이 작동한 것은 로컬 환경에 웹캠이 있었기 때문이다. 그런데 원격 서버에는 웹캠이 없기 때문에 아무런 반응이 없던 것이다. 웹캠이 없는 데스크탑 PC에서 서버를 실행한 것과 같은 것이다. 만약 아마존 원격 서버에 웹캠이 있다고 하더라도 사용자의 웹캠이 아닌 서버의 웹캠이 작동할 것이므로 기존과는 다른 방식을 사용해야 했다. 그래서 JavaScript의 mediadevices.getusermedia() 메서드를 사용하여 사용자의 웹캠을 제어해서 그 프레임을 서버로 보내려는 시도를 해보았다. 결과적으로 사용자의 웹캠을 작동시키는 것은 성공하였으나 프레임을 서버로 전송하여 처리하는 것에는 실패하여 로컬 서버를 사용하는 것으로 할 수 밖에 없었다. 이는 사용자가 라이브러리 설치, 서버 실행 등의 과정을 직접 해야 한다는 것을 의미하므로 치명적인 결함이다.
  
## 소감
처음으로 프로젝트를 수행해 보면서 계획, 분석, 설계, 구현, 테스트 등 소프트웨어 개발의 전과정을 경험해 볼 수 있었다. 지도 교수님께서는 "기획이 없는 구현은 없고, 구현이 없는 기획은 없다"라고 말씀하셨다. 이처럼 기획의 중요성이 구현 못지않다는 것과 개발에 관한 지식이 없이 온전한 기획을 하는 것은 불가능하다는 것을 절실히 느꼈다. 기술적으로는 Django를 사용해 보면서 웹 개발의 흐름을 익힐 수 있었다. 두 학기에 걸쳐 매주 적절한 피드백을 주신 교수님과 열심히 해준 팀원들에게 감사하며 글을 마친다.
