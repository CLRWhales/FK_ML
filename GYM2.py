#setup 
import os, glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import time
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch.nn.functional as F
#net and loader
from Multitask_loader import Multitask_loader
from FK_ver3 import Autoencoder as mod
import numpy as np



#inputs
datapath = 'D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640'
dst = 'C:\\Users\\Calder\\Models'
batch_size = 64
num_workers = 4
LR = 2.5e-4
train_proportion = 0.8
nepochs = 50
patience = 5
min_delta = 1e-7
name = 'FK_ver3_full_transform'
test = False
lambda_recon = 0.5
lambda_reg = 1-lambda_recon

#testing the model on specific files
testlist = [['whale1_1',"D:\\DAS\\FK\\testset\\FK\\20220821T183537Z\\T0_X512_F44_K316_V750.0_20220821T183537Z.png"],
            ['whale1_2',"D:\\DAS\\FK\\testset\\FK\\20220821T183537Z\\T256_X512_F38_K325_V1850.0_20220821T183537Z.png"],
            ['whale2_1',"D:\\DAS\\FK\\testset\\FK\\20220821T183537Z\\T0_X1536_F94_K106_V1550.0_20220821T183537Z.png"],
            ['whale2_2',"D:\\DAS\\FK\\testset\\FK\\20220821T183537Z\\T768_X1536_F86_K118_V1650.0_20220821T183537Z.png"],
            ['nearshore1',"D:\\DAS\\FK\\testset\\FK\\20220821T184417Z\\T0_X0_F4_K268_V350.0_20220821T184417Z.png"],
            ['nearshore2',"D:\\DAS\\FK\\testset\\FK\\20220821T184417Z\\T1280_X0_F7_K29_V4450.0_20220821T184417Z.png"],
            ['FarEnd1',"D:\\DAS\\FK\\testset\\FK\\20220821T183537Z\\T1792_X10240_F6_K483_V150.0_20220821T183537Z.png"],
            ['FarEnd2',"D:\\DAS\\FK\\testset\\FK\\20220821T183537Z\\T2048_X10240_F5_K31_V50.0_20220821T183537Z.png"]
            ]
    

if __name__ == '__main__':
    inputlist = [batch_size, num_workers, LR, train_proportion,nepochs,patience,min_delta,datapath,lambda_recon,lambda_reg]

    tnow = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    outputdir = os.path.join(dst,name + '_' + tnow)
    os.makedirs(outputdir, exist_ok = True)
    txtname = os.path.join(outputdir, 'params.txt')
    with open(txtname, 'w') as f:
        for line in inputlist:
            f.write(f"{line}\n") 

    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5), sigma = 1) #totensor is handled internal to the loader     
    ])

    # Dataset and Dataloaders
    dataset = Multitask_loader(datapath, transform=transform)
    train_size = int(train_proportion * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=num_workers ,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,shuffle=False, pin_memory=True)

    # Model, loss, optimizer 
    model = mod(latent_dim=128).to(device)
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []
    LR_track = []
    counter = 0
    patience_track= []
    TL_reg = []
    TL_recon = []
    VL_reg = []
    VL_recon = []
    full_loss = []


    # Training loop with validation
    for epoch in range(nepochs):
        e_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        reg_RL = 0.0
        recon_RL = 0.0
        for input_img, target_img, reg, in train_loader:
            input_img = input_img.to(device, non_blocking = True)
            target_img = target_img.to(device, non_blocking = True)
            reg = reg.to(device,non_blocking = True)

            optimizer.zero_grad()
            #print(type(input_img), input_img.shape)
            recon,estreg,_ = model(input_img)
            loss_recon = F.mse_loss(recon, target_img)
            loss_reg = F.mse_loss(estreg,reg)
            loss = lambda_recon * loss_recon + lambda_reg * loss_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            full_loss.append(loss.item())
            reg_RL +=loss_reg.item()
            recon_RL +=loss_recon.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        TL_recon.append(recon_RL/len(train_loader))
        TL_reg.append(reg_RL/len(train_loader))
        # Validation phase
        model.eval()
        val_loss = 0.0
        RegVL = 0.0
        ReconVL = 0.0
        with torch.no_grad():
            for input_img, target_img,reg in val_loader:
                input_img = input_img.to(device, non_blocking = True)
                target_img = target_img.to(device, non_blocking = True)
                reg = reg.to(device,non_blocking = True)
                recon,estreg,_ = model(input_img)
                loss_recon = F.mse_loss(recon, target_img)
                loss_reg = F.mse_loss(estreg,reg)
                loss = lambda_recon * loss_recon + lambda_reg * loss_reg
                val_loss += loss.item()
                RegVL +=loss_reg.item()
                ReconVL +=loss_recon.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        VL_reg.append(RegVL/len(val_loader))
        VL_recon.append(ReconVL/len(val_loader))
        eval_losses.append(avg_val_loss)
        LR_track.append(optimizer.param_groups[0]['lr'])


        e_end = time.perf_counter()
        tdif = e_end - e_start

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.9f}, Val Loss = {avg_val_loss:.9f}, Time (s) = {tdif:.1f}")

       
        # Save checkpoint if best
        if avg_val_loss < best_val_loss - min_delta:
            counter = 0
            best_val_loss = avg_val_loss
            pathname = os.path.join(outputdir, 'Epoch' + str(epoch) + '_model.pth')
            torch.save(model.state_dict(), pathname)
            print("✅ Saved new best model.")
        else:
            counter += 1
        
        patience_track.append(counter)
        if counter >= patience:
            print('Ran out of patience >:(, stopping early')
            pathname = os.path.join(outputdir, 'Epoch' + str(epoch) + '_final.pth')
            torch.save(model.state_dict(), pathname)
            print("✅ Saved final model.")
            break
    
        df = pd.DataFrame({'train_loss': train_losses, 'val_loss': eval_losses,'patience':patience_track,'LR':LR_track,'TL_reg':TL_reg,'TL_recon':TL_recon,'VL_reg':VL_reg,'VL_recon':VL_recon})
        fname = os.path.join(outputdir,'loss_log.csv')
        df.to_csv(fname, index=False)

        figure, axis = plt.subplots(3,1)

        axis[0].plot(train_losses, label = 'Training')
        axis[0].plot(eval_losses, label = 'Validation')
        axis[0].plot(TL_recon, label = 'TL_recon')
        axis[0].plot(TL_reg, label = 'TL_reg')
        axis[0].plot(VL_recon, label = 'VL_recon')
        axis[0].plot(VL_reg, label = 'VL_reg')
        axis[0].set_xlabel('epoch')
        axis[0].set_ylabel('MSE')
        axis[0].set_yscale('log')
        axis[0].legend(loc = 'upper right')

        axis[1].plot(patience_track)
        axis[1].set_xlabel('epoch')
        axis[1].set_ylabel('patience')

        axis[2].plot(LR_track)
        axis[2].set_xlabel('epoch')
        axis[2].set_ylabel('Learning rate')

        pname = os.path.join(outputdir,'train_plot.jpeg')
        plt.tight_layout()
        plt.savefig(pname)
        plt.close()

        plt.figure()
        plt.plot(full_loss)
        plt.ylabel('Minibatch loss')
        plt.yscale('log')
        plt.xlabel('minibatch')
        pname = os.path.join(outputdir,'minibatch.jpeg')
        plt.tight_layout()
        plt.savefig(pname)
        plt.close()

    fname = os.path.join(outputdir,'MBloss')
    np.save(fname,full_loss)

    #perfomring opional testing against certain files
    if test:

        t1 = transforms.ToTensor()
        t2 = transforms.GaussianBlur(kernel_size=(5), sigma = 1)
        mpath = glob.glob(os.path.join(outputdir,'*.pth'))[-1]

        testmod = mod().to(device)
        testmod.load_state_dict(torch.load(mpath,weights_only=True))
        testmod.eval()
        testdir = os.path.join(outputdir,'test_img')
        os.makedirs(testdir, exist_ok=True)
        with torch.no_grad():
            for k in testlist:
                image = Image.open(k[1]).convert('L')
                image = t1(image).to(device)
                result,reg,_ = testmod(image.unsqueeze(0))
                #loss = criterion(result, image)

                I = image.cpu().numpy().squeeze()
                R = result.cpu().numpy().squeeze()
                SQD = (I-R)
                
                # Shared color limits for top two
                vmin_shared = min(I.min(), R.min())
                vmax_shared = max(I.max(), R.max())

                #plotting
                fig,ax = plt.subplots(3,1, figsize = (10,10))
                im0 = ax[0].imshow(I,cmap ='gray',vmin = vmin_shared,vmax = vmax_shared)
                ax[0].set_ylabel('Original')
                fig.colorbar(im0,ax = ax[0])
                im1 = ax[1].imshow(R,cmap = 'gray',vmin = vmin_shared,vmax = vmax_shared)
                ax[1].set_ylabel('Reconstruction')
                fig.colorbar(im1,ax = ax[1])
                im2 = ax[2].imshow(SQD, cmap ='seismic', norm = colors.CenteredNorm())
                ax[2].set_ylabel('Difference')
                fig.colorbar(im2, ax = ax[2])
                fname = os.path.join(testdir,k[0] + '.png')
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
                print(k[1])
                print(reg)
