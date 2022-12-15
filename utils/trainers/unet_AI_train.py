def train_unet_AI(model, train_loader, num_epochs):
    optimizer = optim.SGD(G.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(num_epochs):
        losses = []
        epoch_start_time = time.time()
        batch_idx = 0
        for data in tqdm(train_loader):
            input = data[0].to(device=device) # images
            
            if (model.n_channels == 4):
                depth = data[3].to(device=device)
                input = torch.cat([input, depth], 1)
            
            target = data[1].to(device=device) # sementic segmentations
            target = convert_to_one_hot(torch.round(target * 12), model.n_classes)
            model.zero_grad()
            optimizer.zero_grad()
            output = model(input)
            loss = F.binary_cross_entropy_with_logits(output, target)
            
            loss.backward()
            optimizer.step()
            
            _loss = loss.detach().item()
            losses.append(_loss)
            batch_idx += 1
            
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - using time: %.2f seconds' % ((epoch + 1), num_epochs, per_epoch_ptime))
        print('loss of generator G: %.3f' % (torch.mean(torch.FloatTensor(losses))))
        if epoch == 0 or (epoch + 1) % 5 == 0:
            with torch.no_grad():
                if (model.n_channels == 4):
                    show_result(model, fixed_x_D, fixed_y_D, (epoch+1))
                else:
                    show_result(model, fixed_x, fixed_y, (epoch+1))
        
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.binary_cross_entropy_with_logits(output, target, size_average=False).item() # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))