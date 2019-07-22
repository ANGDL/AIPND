# Github关联本地仓教程

## 一、 注册github（略过）

## 二、 创建新的repository

### 2.1 点击右上角的头像，出现如下下拉框，点击【Your repositories】

![Screenshot-9.png](file:///C:/Users/ANG_Z/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)



### 2.2 在如下的界面中点击【new】按钮

![Screenshot-10.png](file:///C:/Users/ANG_Z/AppData/Local/Temp/msohtmlclip1/01/clip_image003.png)  

### 2.3 接着填写该repository的相关信息

![Screenshot-11.png](file:///C:/Users/ANG_Z/AppData/Local/Temp/msohtmlclip1/01/clip_image005.png)  

### 2.4 创建成功后出现以下界面，关注下显示的命令，有用

![Screenshot-12.png](file:///C:/Users/ANG_Z/AppData/Local/Temp/msohtmlclip1/01/clip_image007.png)

![img](file:///C:/Users/ANG_Z/AppData/Local/Temp/msohtmlclip1/01/clip_image009.jpg)

##  三、安装git客户端（如果已经安装、跳到下一步）

下载地址：<https://git-scm.com/downloads>

对应自己的系统进行选择

![img](file:///C:/Users/ANG_Z/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png)

## 四、 添加公钥到github

### 4.1 Windows下打开Git Bash（Mac 打开终端），创建SSH Key：

```
$ ssh-keygen -t rsa -C "youremail@example.com"
```

你需要把邮件地址换成你自己的邮件地址，然后一路回车，使用默认值即可，由于这个Key也不是用于军事目的，所以也无需设置密码。

![Screenshot-4.png](https://tcs.teambition.net/thumbnail/111c34fdf08380ed5fee24b6e4bd9b1049eb/w/398/h/144)

如果一切顺利的话，可以在用户主目录里找到`.ssh`目录，里面有`id_rsa`和`id_rsa.pub`两个文件，这两个就是SSH Key的秘钥对，`id_rsa`是私钥，不能泄露出去，`id_rsa.pub`是公钥，可以放心地告诉任何人。

![Screenshot-5.png](https://tcs.teambition.net/thumbnail/111c8b3444384a6118b44362f21387d042f2/w/600/h/108)  

### 4.2 登陆GitHub，打开“Account settings”，“SSH Keys”页面：

然后，点“Add SSH Key”，填上任意Title，在Key文本框里粘贴`id_rsa.pub`文件的内容：

![Screenshot-7.png](https://tcs.teambition.net/thumbnail/111c039aac4afd23d2a0298a6c6d3d3087dc/w/799/h/378)  

点“Add Key”，你就应该看到已经添加的Key：

![Screenshot-8.png](https://tcs.teambition.net/thumbnail/111c6a3328b6ef2011c065fc1015251d1ad8/w/600/h/282)

## 五、使用git init 命令初始化项目 

如果是Windows用户， 打开git bash 命令窗口，如果是mac用户，打开终端命令窗口。

输入：`git init`

完成这个步骤后，项目文件夹下会出现.git的文件夹，包含了改项目的git版本信息、修改、提交等记录文件。

## 六、添加关联github远程仓库

在命令行中输入以下命令（可以直接复制2.4中显示的命令）：

`git remote add origin git@github.com:ANGDL/ML.git`

`git@github.com:ANGDL/ML.git`是替换成自己的repository地址

完成以上步骤后，已经全部关联完成。以后每次修改项目，可以使用`git push -u origin master`同步到github。