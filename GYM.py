#setup 
import os, glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import time
from torchvision.transforms import functional as F
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torchsummary import summary

#net and loader
from MaskedImageDataset import MaskedImageDataset
from DenoDAS import UNET as mod



#cutom crop to reove nyquist, can get rid on later iterations
# class RemoveBottomRow:
#     def __call__(self, img):
#         w, h = img.size  # PIL Image: (width, height)
#         return F.crop(img, top=0, left=0, height=h - 1, width=w)



#inputs
datapath = 'D:\\DAS\\FK\\NoShore_DS3_x512_20250604T143445'
batch_size = 64
num_workers = 4
LR = 1e-3
mask_prob = 0.0
test_proportion = 0.8
nepochs = 100
patience = 5
min_delta = 1e-5
name = 'UNET'
test = False

#testing the model on specific files
testlist = [['whale1_1',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T183537Z\\FK256_T0_X768_20220821T183537Z.png"],
            ['whale1_2',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T183537Z\\FK256_T0_X1024_20220821T183537Z.png"],
            ['whale1_3',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T183537Z\\FK256_T0_X1280_20220821T183537Z.png"],
            ['whale2_1',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T183537Z\\FK256_T0_X1792_20220821T183537Z.png"],
            ['whale2_2',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T183537Z\\FK256_T0_X2048_20220821T183537Z.png"],
            ['onshore1',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T181007Z\\FK256_T512_X0_20220821T181007Z.png"],
            ['onshore2',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T184337Z\\FK256_T0_X0_20220821T184337Z.png"],
            ['nearshore1',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T184337Z\\FK256_T0_X256_20220821T184337Z.png"],
            ['nearshore2',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T184057Z\\FK256_T0_X256_20220821T184057Z.png"],
            ['FarEnd1',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T184057Z\\FK256_T256_X10496_20220821T184057Z.png"],
            ['FarEnd2',"D:\\DAS\\FK\\DS3_x512_20250530T153412\\FK\\20220821T181317Z\\FK256_T1536_X10496_20220821T181317Z.png"]
            ]
    

if __name__ == '__main__':
    
    
    inputlist = [batch_size, num_workers, LR, mask_prob, test_proportion,nepochs,patience,min_delta]

    tnow = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    outputdir = os.path.join(datapath,name + '_' + tnow)
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
        #RemoveBottomRow(), #goes from 257x512 to 256x512 removing the nyquist row
        transforms.ToTensor() #this also scales uint 8 inputs to range 0-1
        #transforms.Normalize(mean = [0.0],std = [1.0])
    ])

    # Dataset and Dataloaders
    dataset = MaskedImageDataset(datapath, transform=transform, mask_prob=mask_prob)
    train_size = int(test_proportion * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=num_workers ,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,shuffle=False, pin_memory=True)

    # Model, loss, optimizer 
    model = mod().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.

    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []
    counter = 0
    patience_track= []
    # Training loop with validation
    for epoch in range(nepochs):
        e_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for input_img, target_img in train_loader:
            input_img = input_img.to(device, non_blocking = True)
            target_img = target_img.to(device, non_blocking = True)

            optimizer.zero_grad()
            #print(type(input_img), input_img.shape)
            output = model(input_img)
            loss = criterion(output, target_img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_img, target_img in val_loader:
                input_img, target_img = input_img.to(device, non_blocking = True), target_img.to(device, non_blocking = True)
                output = model(input_img)
                loss = criterion(output, target_img)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        eval_losses.append(avg_val_loss)

        e_end = time.perf_counter()
        tdif = e_end - e_start

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time (s) = {tdif:.4f}")

       
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
    
    df = pd.DataFrame({'train_loss': train_losses, 'val_loss': eval_losses,'patience':patience_track})
    fname = os.path.join(outputdir,'loss_log.csv')
    df.to_csv(fname, index=False)

    figure, axis = plt.subplots(2,1)

    axis[0].plot(train_losses, label = 'Training')
    axis[0].plot(eval_losses, label = 'Validation')
    axis[0].set_xlabel('epoch')
    axis[0].set_ylabel('MSE')
    axis[0].set_yscale('log')
    axis[0].legend(loc = 'upper right')

    axis[1].plot(patience_track)
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('patience')
    pname = os.path.join(outputdir,'train_plot.jpeg')
    plt.tight_layout()
    plt.savefig(pname)
    plt.close()

    #perfomring opional testing against certain files
    if test:
        mpath = glob.glob(os.path.join(outputdir,'*.pth'))[-1]

        testmod = mod().to(device)
        testmod.load_state_dict(torch.load(mpath,weights_only=True))
        testmod.eval()
        testdir = os.path.join(outputdir,'test_img')
        os.makedirs(testdir, exist_ok=True)
        with torch.no_grad():
            for k in testlist:
                image = Image.open(k[1]).convert('L')
                image = transform(image).to(device)
                result = testmod(image.unsqueeze(0))
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
                fname = os.path.join(testdir,k[0] + '.jpeg')
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
