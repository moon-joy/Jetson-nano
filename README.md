# 제목
#### 소제목

*목록1
+목록2
-목록3
 +하위목록

 > 인용
*** (구분선, 실제 작성시엔 괄호내 삭제)
```코드``` (~ 버튼에 있는 ` 이다   ' 아님 )

잿슨나노의 기본세팅이 되어있다는 전제하에 진행하도록 하겠습니다.
기본세팅의 자세한 사항은 https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write 에서 확인해주세요.

""참고 링크 https://github.com/jetsonmom/yolov8_jetson4GB?tab=readme-ov-file   ,    https://cyb.tw/docs/Tech/2020/9/18_Install-anaconda-on-Jetson-Nano.html#install-archiconda ""

sudo apt update
sudo apt upgarde
wget --quiet -O archiconda.sh https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh && \
    sh archiconda.sh -b -p $HOME/archiconda3 && \
    rm archiconda.sh
export PATH=$HOME/archiconda3/bin:$PATH
conda config --add channels gaiar && \
conda config --add channels conda-forge && \
conda config --add channels c4aarch64 && \
conda update -n base --all && \
conda install -y python=3.8 libiconv && \
conda install -y conda-build && \
conda install -y anaconda-client
uname -a
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
sudo chmod 755 Archiconda3-0.2.3-Linux-aarch64.sh
