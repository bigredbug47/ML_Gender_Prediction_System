#FUNCTION: RENAME MULTIPLE FILE
def rename_file(path,name):
    import shutil
    import glob
    path = path+"/"+name
    i = 1
    for filename in glob.glob(os.path.join(path, '*.png')):
        print(filename)
        new_name=path+"/"+name+"_"+str(i)+".png"
        print(new_name)
        shutil.move(filename, new_name)
        i=i+1


#FUNCTION: RANDOM TRAINING DATA
def random_file(path_file):
    import random
    num_rand = random.sample(range(1,72),71)
    #MALE
    file_train_txt = open(path_file+"/train.txt","w") 
    file_train_lb = open(path_file+"/lbtrain.txt","w")
    file_test_txt = open(path_file+"/test.txt","w") 
    file_test_lb = open(path_file+"/lbtest.txt","w")
    for i in num_rand[0:57]:
        file_train_txt.write("male_"+str(i)+"\n")
        file_train_lb.write("1\n")
    for i in num_rand[57:71]:
        file_test_txt.write("male_"+str(i)+"\n")
        file_test_lb.write("1\n")
    file_train_txt.close() 
    file_train_lb.close() 
    file_test_txt.close() 
    file_test_lb.close()

    #FEMALE
    num_rand = random.sample(range(1,88),87)
    file_train_txt = open(path_file+"/train.txt","a") 
    file_train_lb = open(path_file+"/lbtrain.txt","a")
    file_test_txt = open(path_file+"/test.txt","a") 
    file_test_lb = open(path_file+"/lbtest.txt","a")
    for i in num_rand[0:70]:
        file_train_txt.write("female_"+str(i)+"\n")
        file_train_lb.write("0\n")
    for i in num_rand[70:88]:
        file_test_txt.write("female_"+str(i)+"\n")
        file_test_lb.write("0\n")
    file_train_txt.close() 
    file_train_lb.close() 
    file_test_txt.close() 
    file_test_lb.close()
    print("DONE WRITE FILE")

#FUNCTION: EXTRACT FEATURE VGG16
def extract_fea_vgg16(path_fea,path_img):
    from vgg16 import VGG16
    from keras.preprocessing import image
    from keras.models import Model
    from imagenet_utils import preprocess_input
    import numpy as np
    import glob
    import os
    vgg16model = VGG16(weights='imagenet', include_top=True);
    model = Model(inputs=vgg16model.input, outputs=vgg16model.get_layer('fc2').output)

    for filename in glob.glob(os.path.join(path_img, '*.png')):
        name_save=(filename.replace(path_img,"")).replace(".png","")
        file_feature = open(path_fea+"/"+name_save+".mat","wb") 
        img_path=filename
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        np.savetxt(file_feature, features)
        file_feature.close()
        print("Step done")
    print("DONE EXTRACT FEATURE")

    
#FUNCTION: INITIALIZE INPUT
def initialize_input(path_training,path_features,file_name):
    import os
    import numpy as np
    input_file = open(path_training + "/"+file_name,"r")
    if file_name == "test.txt" or file_name == "train.txt":
        arr_training = []
        for line in input_file.readlines():
            for i in line.split():
                temp=[]
                inp=open(path_features + "/" + str(i) + ".mat","r")
                for subline in inp.readlines():
                # loop over the elemets, split by whitespace
                    for subi in subline.split():
                        # convert to float and append to the list
                        temp.append(float(subi))
                if len(arr_training) == 0:
                    arr_training.append(temp)
                else:
                    arr_training=np.vstack([arr_training,temp]) 
                inp.close()
        input_file.close()
        return arr_training
    elif file_name == "lbtest.txt" or file_name == "lbtrain.txt":
        arr_lbtest=[]
        temp=[]
        for line in input_file.readlines():
            for i in line.split():
                if len(arr_lbtest) == 0:
                    temp.append(int(i))
                    arr_lbtest.append(temp)
                else:
                    temp=[]
                    temp.append(int(i))
                    arr_lbtest=np.vstack([arr_lbtest,temp])
        input_file.close()
        return arr_lbtest

#FUNCTION: MERGE DATA TO 1 MATRIX
def get_training_input(path_training,path_features):
    arr_test = initialize_input(path_training,path_features,"test.txt")
    arr_lbtest = initialize_input(path_training,path_features,"lbtest.txt")
    arr_train = initialize_input(path_training,path_features,"train.txt")
    arr_lbtrain = initialize_input(path_training,path_features,"lbtrain.txt")
    return [
        arr_test,
        arr_lbtest,
        arr_train,
        arr_lbtrain
    ]

#FUNCTION: LINEAR REGRESSION MODEL
def sklearn_linear_model(x_train,lbx_train,x_test,lbx_test):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, lbx_train)
    
    # Make predictions using the testing set
    x_pred = regr.predict(x_test)

    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(lbx_test, x_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(lbx_test, x_pred))


#FUNCTION: SVM MODEL
def sklearn_svm(x_train,lbx_train,x_test,lbx_test,kernel_type,kernel_input):
    from sklearn import svm
    from sklearn.metrics import mean_squared_error, r2_score
    if kernel_type == "nonlinear":
        clf = svm.NuSVC()
        clf.fit(x_train, lbx_train)
        x_pred = clf.predict(x_test)
         # The mean squared error
        print("NuSVC Kernel :")
        print("Mean squared error: %.2f"
              % mean_squared_error(lbx_test, x_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(lbx_test, x_pred))
    else:
        clf = svm.SVC(kernel=kernel_input)
        clf.fit(x_train, lbx_train)
        x_pred = clf.predict(x_test)
         # The mean squared error
        print("SVC Kernel " + kernel_input + " :")
        print("Mean squared error: %.2f"
              % mean_squared_error(lbx_test, x_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(lbx_test, x_pred))


#MAIN 
my_own_path="D:/CAMI/DIP/MLinCV"
path_training = my_own_path+"/training_data_002/db/db3"
#random_file(path_training)
path_fea = my_own_path+"/training_data_002/features/vgg16"

#Parameters have 4 elements: 
#         arr_test(X,Y) - X: number of samples, Y: features in sample (n,m)
#         arr_lbtest(X,Y) - X:number of samples, Y: Label of sample (n,1)
#         arr_train(X,Y)
#         arr_lbtrain(X,Y)
parameters = get_training_input(path_training,path_fea)

x_test,lbx_test,x_train,lbx_train = parameters

sklearn_linear_model(x_train,lbx_train,x_test,lbx_test)


# sklearn_svm(x_train,lbx_train,x_test,lbx_test,"linear","linear")
# sklearn_svm(x_train,lbx_train,x_test,lbx_test,"linear","rbf")
# sklearn_svm(x_train,lbx_train,x_test,lbx_test,"linear","sigmoid")
# sklearn_svm(x_train,lbx_train,x_test,lbx_test,"nonlinear","linear")

print("DONE")