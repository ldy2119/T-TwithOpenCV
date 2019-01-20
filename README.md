# 설명
OpenCV로 Tooth and Tail을 플레이하는 프로젝트입니다.

# 구현해야 할 기능
~~1. 보유하고 있는 식량 인식~~  
2. 맵 인식  
3. 움직이기  
4. 미니맵 인식  
5. 마우스 클릭으로 적을 공격하기  
6. 농장과 병영 

# 날짜

## 0일차
OpenCV 분석 및 계획 수립

## 1일차
OpenCV로 식량 인식을 구현하려고 노력했지만 라이브러리를 제대로 사용하지 못 해 실패함.

## 2일차
Python으로 해보기로 결정. 아직 제대로 인식이 되지 않음. 머신 러닝 같은 걸 해야 하나?
내일은 가족여행 때문에 못 할 것 같음.

## ~~3일차~~
~~쉬었음~~ 날짜를 잘못 계산한 것 같다. 아마 이날도 코딩한 듯.

## 3일차
라이브러리 교체, 식량 인식 기능 구현 대부분 완료(제대로 검출되지 않는 부분이 있음. 테스트 필요)

## 4일차
지도 인식을 하는 중. 일단 특정 맵에서만 테스트 해보려고 함.(특정 상황에서만 동작하도록 만들 예정)

## 5일차
코드 상으로는 진행이 없지만 테스트 시나리오는 작성 중. 벽을 따라 움직이도록 할 예정이다.

## 6일차
코딩하다 뻗어서 업로드를 못 했다. 7일차 즈음에 벽을 따라 움직이도록 할 예정이다.

## 7일차
이제 벽을 인식하는 것 같다. 벽을 따라 이동할 수 있다.
1월 6일, 7일, 8일은 놀러가서 커밋 안 할 예정.

## ~~8, 9, 10일차~~
강원도 여행 갔다 왔다.

## ~~11일차~~
후유증으로 뻗어있었다.

## 12일차
지도에서 풍차의 위치를 인식했다.

## 13일차
지도에서 농장의 위치를 파악했다. 농장을 건설하는 건 나중에. 통합 언제 하냐;;

## 14일차
마우스 클릭을 구현했다. 진짜 통합 언제 할 수 있지?

## ~~15일차~~
으악 12시 지남

## 16일차
인식이 잘 안 돼서 맵을 쪼개서 분석하기로 했다. 메모리와 CPU가 버텨준다면 이 방법이 나을 것 같다.<br>
시간이 없다;; 예전에 벽을 따라 이동하는 걸 구현했다고 했는데 안 한 듯. 구현 중입니다.

## 17일차
벽을 따라 이동하는 것 대신 직접 이동하기로 했다.
정밀한 움직임이 보장되지 않는 점이 수정 요소.

## 18일차
이때까지 벽을 따라 이동하려고 했는데 그걸 다 때려치고 알고리즘을 새로 짜려니 힘들긴 한데 더 쉬운 것 같다..?<br>
아무튼 test14.py 진행 중. 다음 수정 요소는 제대로 동작하도록 수정하기와 설명 적기.

## 19일차
맞게 하고 있는건지 모르겠다. 아직 경로를 만들어주는 알고리즘을 짜지 못 해 경로를 따라 이동하는 것 자체가 불가능하다.<br>
그래도 놀지만 않으면 내일이나 모레쯤 끝날 것 같다.

## 20일차
테스트를 제외하고 전부 끝냈다. 몸이 힘드니 코딩이 점점 느려지는 것 같아 슬프다.<br>
"캐릭터가 이 좌표로 가면 벽에 안 부딪힙니다" 라는 가이드라인은 만들었으니 이제 캐릭터가 갈 수 있는 올바른 경로를 설정하고 테스트하면 될 것 같다.

## 21일차
테스트해보니 제대로 동작하지 않는 부분이 있다. 올바른 경로를 만들기 위해서는 순서를 맞춰 줘야 하는데 그 부분도 잘 되지 않는 것 같다.

## 22일차
편법을 이용해서 작동은 하게 만들었다. 이제 많은 케이스를 테스트해봐야 하는데 지친다.<br>
다른 기능은 또 언제 구현할까...
