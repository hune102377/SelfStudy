# 모듈 로딩
import cv2
import time
import math
import ffmpeg
import os
import re
from moviepy.editor import *
from yolov5facedetector.face_detector import YoloDetector

class Deidentification:
    def __init__(self, url=0, output_save=False):
        ''' *********************** Initialize *********************** 
        - output_save : 결과물 저장 여부
                - True : 저장
                - Fasle : 저장하지 않음
        '''
        self.url=url
        self.isDragging=False
        self.x0, self.y0, self.w, self.h=-1, -1, -1, -1
        self.output_save=output_save


    def onMouse(self, event, x, y, flags, param):
        """
        ROI 영역 지정

        *** parameters ***
        event : 마우스 이벤트
        x, y : ROI 영역의 꼭짓점 좌표
        """
        blue, red = (255, 0, 0), (0, 0, 255)

        if event==cv2.EVENT_LBUTTONDOWN:
            self.isDragging=True
            self.x0, self.y0=x, y

        elif event==cv2.EVENT_MOUSEMOVE:
            if self.isDragging:
                cv2.rectangle(self.frame, (self.x0, self.y0), (x, y), blue, 2)

        elif event==cv2.EVENT_LBUTTONUP:
            if self.isDragging:
                self.isDragging=False
                self.w=x-self.x0
                self.h=y-self.y0

                if self.w>0 and self.h>0:
                    cv2.rectangle(self.frame, (self.x0, self.y0), (x, y), red, 2)

                else:
                    print('drag should start from left-top side')


    def roi_inside_mosaic(self):
        '''
        ROI 영역 내 얼굴 모자이크

        x0, y0, w, h는 onMouse 함수에서 설정된 값
        ori_frame_copy, frame은 mosaic 함수에서 설정된 값
        ''' 
        if (self.x0>0) and (self.y0>0) and (self.w>0) and (self.h>0):
            self.ori_frame_copy[int(self.y0):int(self.y0+self.h), int(self.x0):int(self.x0+self.w)]=self.frame[int(self.y0):int(self.y0+self.h), int(self.x0):int(self.x0+self.w)]
            self.frame=self.ori_frame_copy
            cv2.rectangle(self.frame, (int(self.x0), int(self.y0)), (int(self.x0+self.w), int(self.y0+self.h)), (0, 255, 0), 2, 1)


    def roi_outside_mosaic(self):
        '''
        ROI 영역 외 얼굴 모자이크
        '''
        if (self.x0>0) and (self.y0>0) and (self.w>0) and (self.h>0):
            self.frame[int(self.y0):int(self.y0+self.h), int(self.x0):int(self.x0+self.w)] = self.ori_frame_copy[int(self.y0):int(self.y0+self.h), int(self.x0):int(self.x0+self.w)]
            cv2.rectangle(self.frame, (int(self.x0), int(self.y0)), (int(self.x0+self.w), int(self.y0+self.h)), (0, 255, 0), 2, 1)
        

    def video_save(self, webcam, file_name):
        '''
        영상 저장을 위한 객체 생성
        '''
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = round(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = webcam.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))   # 파일명 경로
        return fps, out


    def compact_save(self, PATH='./', crf=25):
        '''
        모자이크 영상 용량 줄이는 함수

        *** parameters ***
        crf : 낮을수록 고화질이지만, 용량 높아짐
        '''
        self.sound()
            
        # 모자이크 용량 줄이기
        try:
            print('\n\n\ndslkfjfkj', os.listdir())
            os.remove(self.video_names)  # 음성 없는 모자이크 영상 삭제
            os.remove(self.video_names.split('_mosaic')[0]+'.mp3')  # 원본 음성 삭제
        
        except: pass

        self.video_names = 'end_'+self.video_names
        
        if os.path.isdir('resized')==False :
            # ffmpeg으로 변환할 때는 기존 파일에 덮어쓰기가 안되므로 변환된 파일들을 저장할 폴더를 생성
            os.mkdir('resized')

        # 변환 중 에러가 발생하는 경우가 있어 에러 처리를 위한 리스트를 만듦
        err = []

        # (try) 파일 변환을 시도함.
        # (except) 에러가 날 경우 변환 중이던 파일을 지우고 파일명을 err 리스트에 추가함
        try:
            # crf[quality] : 비트레이트 대신 화질 기준으로 인코딩할 때 쓰는 옵션. libx264 코덱 기준 사용 가능 범위 0-51, 0은 무손실, 디폴트는 23
            ffmpeg.input(PATH+self.video_names).output('resized/'+self.video_names, crf=crf, vsync='vfr').run()

        except:
            os.remove('resized/'+self.video_names)
            err.append(self.video_names)


        # 파일이 resized 폴더 내에 있다면(에러가 나지 않고 변환되어 옮겨져 있다면) 원래 폴더에서 파일을 삭제함
        if self.video_names in os.listdir('resized'):
            os.remove(PATH+self.video_names)


        # mp4 파일 목록을 순화하면서 오류 났던 파일 (err 리스트)을 제외한 나머지 파일을 원래 폴더로 모두 옮김
        if self.video_names in err:
            pass
        else:
            os.replace('resized/'+self.video_names, PATH+self.video_names)

        # resized 폴더를 삭제함
        os.rmdir('resized')

        # 변환에 실패한 파일 목록을 출력함
        # print('실패한 목록: \n', err)


    def sound(self):
        '''
        모자이크 영상에 음성 추출 함수
        '''
        my_path='./yolov5/'

        print('sound 함수에서 video_names =>', self.video_names)
        try:
            print('try로 들어왔다')
            print('aa => ', self.video_names.split('_mosaic')[0]+'.mp4')
            # 원본 영상에서 음성 추출
            ffmpeg.input(self.video_names.split('_mosaic')[0]+'.mp4').output(self.video_names.split('_mosaic')[0]+'.mp3').run()
            print('첫번째')

            # 모자이크 영상과 원본 음성 합치기
            videoclip = VideoFileClip(self.video_names)
            print('두번째')
            audioclip = AudioFileClip(self.video_names.split('_mosaic')[0]+'.mp3')
            print('세번째')

            videoclip.audio = audioclip
            print('네번째')
            videoclip.write_videofile('end_'+self.video_names)
            print('다섯번째')

        except:
            print('except로 들어왔다')
            os.rename(self.video_names, 'end_'+self.video_names)


    def mosaic(self, yolo_type='yolov5n', target_size=1080, gpu=0, min_face=0, conf_thres=0.3, iou_thres=0.5, roi='inside', sigma=55):
        '''
        모자이크 하는 함수

        *** parameters ***
        - yolo_type : yolov5 모델 버전
                    - yolov5n (default)
                    - yolov5m
                    - yolov5l
        - gpu : gpu number (int) or -1 or string for cpu.
        - min_face : minimal face size in pixels.
        - target_size : target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080.
                        None for original resolution.
        - frontal : if True tries to filter nonfrontal faces by keypoints location.
        - conf_thres: confidence threshold for each prediction
        - iou_thres: threshold for NMS (filtering of intersecting bboxes)
        - roi : ROI 영역 내/외 얼굴 모자이크
                - inside : ROI 영역 내 얼굴 모자이크
                - outside : ROI 영역 외 얼굴 모자이크
        - sigma : 흐림 강도 조절
        '''
        model = YoloDetector(yolo_type=yolo_type, target_size=target_size, gpu=gpu, min_face=min_face)
        webcam = cv2.VideoCapture(self.url)

        if '\\' in self.url:
            self.video_name = re.sub('.mp4|.avi','', self.url.split('\\')[-1])
        else:
            self.video_name = re.sub('.mp4|.avi','', self.url.split('/')[-1])

        self.video_names=f"{self.video_name}_mosaic.mp4"
        print('mosaic에서 video_names =>', self.video_names)

        if self.url==0:
            fps, out=self.video_save(webcam, 'mosaic.mp4')
        else:
            fps, out=self.video_save(webcam, self.video_names)
        
        # 동영상 파일 열기 성공 여부 확인
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()
            
        mosa=True          # mosa=True일 때 영상의 기본 상태 : 모자이크
        fps_value=1./fps   # 원본 fps와 맞추기 위해 나누어 줄 숫자
        gauge_bar=-1       # 게이지 바 초기화

        f = open('./test.txt', 'w')  # 파일 열기
        f.write('frame_num\t\t\t\tbboxes\t\t\tpeople_num\t\t\t\ttime\t\t\t\t\tframe\n')  # 파일 컬럼명
        f = open('./test.txt', 'a')  # 파일 열기
        
        while webcam.isOpened():
            status, self.frame = webcam.read()
            #print('\n\nfps_value =>', fps_value, '\n')

            # 알고리즘 시작 지점
            start_time=time.time()

            if self.url==0: self.frame=cv2.flip(self.frame, 1)   # 웹캠일 때 영상 좌우 반전

            if self.frame is None: break   # 영상이 끝났을 때 빠져나감

            self.ori_frame_copy=self.frame.copy()
            

            # 영상 처리 속도 개선
            if (webcam.get(cv2.CAP_PROP_POS_FRAMES) == 1) | (webcam.get(cv2.CAP_PROP_POS_FRAMES) == 2) | (webcam.get(cv2.CAP_PROP_POS_FRAMES) % fps_value == 0):
                bboxes, _, _ = model.predict(self.frame, conf_thres=conf_thres, iou_thres=iou_thres)

                for idx in range(len(bboxes[0])):
                    f.write('%-3d \t\t\t' %int(webcam.get(cv2.CAP_PROP_POS_FRAMES)))  # frame_num
                    f.write('%-22s \t\t' %bboxes[0][idx])  # bboxes
                    f.write('%-3d \t\t' %len(bboxes[0]))  # people_num
                    f.write("%12s \t\t" %time.strftime('%Y-%m-%d %H:%M:%S'))  # time
                    f.write('%s \n' %self.frame.flatten())  # frame   # 1차원으로 바꾼 거라서 사용하려면 형태 reshape 해야 함

            else: pass

            if not status:
                print("Could not read frame")
                exit()
                
            key = cv2.waitKey(1)
            if key == 26: # Ctrl + Z :  모자이크 켜짐
                mosa = True
            elif key == 24: # Ctrl + X :  모자이크 꺼짐
                mosa = False
        
            for bbox in bboxes[0]:
                (startX, startY)=bbox[0], bbox[1]
                (endX, endY)=bbox[2], bbox[3]

                if mosa == True:
                    face_region = self.frame[startY:endY, startX:endX] # 관심영역(얼굴) 지정
                    
                    # 모자이크
                    '''
                    cv2.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None) -> dst

                    • src: 입력 영상. 각 채널 별로 처리됨.
                    • dst: 출력 영상. src와 같은 크기, 같은 타입.
                    • ksize: 가우시안 커널 크기. (0, 0)을 지정하면 sigma 값에 의해 자동 결정됨
                    • sigmaX: x방향 sigma.
                    • sigmaY: y방향 sigma. 0이면 sigmaX와 같게 설정.
                    • borderType: 가장자리 픽셀 확장 방식.
                    '''
                    self.frame[startY:endY, startX:endX] = cv2.GaussianBlur(face_region, ksize=(0,0), sigmaX=sigma)


            # ROI
            if roi=='inside':
                self.roi_inside_mosaic()   # ROI 영역 내 얼굴 모자이크
            elif roi=='outside':
                self.roi_outside_mosaic()   # ROI 영역 밖 얼굴 모자이크
            else: print('roi should be either inside or outside. Please enter the correct one.')
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # 'q'키 입력 받으면 윈도우 창이 종료
                break
                    
            # display output
            if self.output_save==True:
                out.write(self.frame) # 동영상 저장

            cv2.imshow('Mosaic Video', self.frame) # 윈도우 창에 이미지를 띄움
            cv2.setMouseCallback("Mosaic Video", self.onMouse)  # 창 이름 바로 윗줄 창 이름과 같아야함
            

            if webcam.get(cv2.CAP_PROP_POS_FRAMES) == 2:
                fps_value = math.ceil((time.time() - start_time)/(1./fps))

            if gauge_bar != round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT)):

                print(' '*round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT))+'▽',\
                    round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT)),'%')
                print('[', end='')
                print(round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT))*'■'+\
                    (100-round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT)))*' ', end='')
                print(']')
                gauge_bar = round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT))
            
        f.close()   # 파일 닫기

        # 동영상 파일을 닫고 메모리 해제
        out.release()
        webcam.release()
        
        # 모든 윈도우 창을 닫음
        cv2.destroyAllWindows()
        

if __name__=='__main__':
    #model=Deidentification(url=r'C:\Users\USER\Desktop\my_yolo5\yolov5\dy_test\test_01.mp4', output_save=True)
    #model=Deidentification(url=r'C:\Users\USER\Desktop\my_yolo5\yolov5\mosaic_result\kampus.mp4', output_save=True)
    #model=Deidentification(url=r'C:\Users\USER\Desktop\my_yolo5\yolov5\people_test\people.mp4', output_save=True)
    model=Deidentification(url=r'C:\Users\USER\Desktop\my_yolo5\yolov5\people_test\people_street.mp4', output_save=True)
    #model.mosaic(roi='outside', sigma=20)
    model.mosaic(sigma=20, roi='outside')

    if model.output_save==True:
        model.compact_save()
        
    print('Done.')


