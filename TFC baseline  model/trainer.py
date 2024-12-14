import numpy as np
import os
import math
import sys
from sklearn.metrics import confusion_matrix

sys.path.append("..")

from TFC.loss import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from TFC.model import *
from torch.utils.data import TensorDataset, DataLoader
from downstream_tasks.classification import eval_classification


def linear_train(model, optimizer, train_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    total_clf_loss = []

    for epoch in range(config.num_epoch):

        for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
            model.train()
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)
            # aug1 = aug1.float().to(device)
            # aug1_f = aug1_f.float().to(device)

            prediction = model(data, data_f)
            clf_loss = criterion(prediction, labels)
            clf_loss.backward()
            optimizer.step()
            total_clf_loss.append(clf_loss.item())

        clf_loss_epoch = torch.tensor(total_clf_loss).mean().item()
        print(f"Epoch number: {epoch}, clf_loss: {clf_loss_epoch}")


def linear_test(model, test_loader, config, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    performance = {}

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _, data_f, _ in test_loader:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            prediction = model(data, data_f)
            clf_loss = criterion(prediction, labels)
            acc_bs = labels.eq(prediction.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels, num_classes=config.num_classes)
            pred_numpy = prediction.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,
                                       average="macro", multi_class="ovr")
            except:
                auc_bs = np.float(0)
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
            if math.isnan(prc_bs):
                prc_bs = np.float(0)
            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_acc.append(acc_bs)
            total_auc.append(auc_bs)
            total_prc.append(prc_bs)

            total_loss.append(clf_loss.item())
            pred = prediction.max(1, keepdim=True)[1]  # get the index of the max log-probability
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

            matrix = confusion_matrix(pred.detach().cpu().numpy(), labels.data.cpu().numpy())
            print(matrix)

        labels_numpy_all = labels_numpy_all[1:]
        pred_numpy_all = pred_numpy_all[1:]

        # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
        # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
        precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
        recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
        F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
        acc = accuracy_score(labels_numpy_all, pred_numpy_all)

        total_loss = torch.tensor(total_loss).mean()
        total_auc = torch.tensor(total_auc).mean()
        total_prc = torch.tensor(total_prc).mean()

        performance['Accuracy'] = acc
        performance['Precision'] = precision
        performance['Reall'] = recall
        performance['F1'] = F1
        performance['AUROC'] = total_auc.item()
        performance['AUPRC'] = total_prc.item()
        """print('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f' % (
        acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100))"""
        return performance


def model_pretrain(model, model_optimizer, criterion, train_loader, config, device):
    total_loss = []
    model.train()
    global loss, loss_t, loss_f, l_TF, loss_c, data_test, data_f_test

    # optimizer
    model_optimizer.zero_grad()

    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1 = aug1.float().to(device)
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)  # device, 128, 0.2, True

        # print('h_t shape:', h_t.shape)
        # print('h_t_aug shape:', h_t_aug.shape)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)  # this is the initial version of TF loss

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug,
                                                                                                            z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam * (loss_t + loss_f) + l_TF

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    print('Pretraining: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss, loss_t, loss_f, l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    return ave_loss


def pretraining(model, model_optimizer, train_dl, device, config, training_mode='pre_train', seed=42):
    # Start training
    print("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if training_mode == 'pre_train':
        print('Pretraining on source dataset')
        for epoch in range(1, config.num_epoch + 1):
            # Train and validate
            """Train. In fine-tuning, this part is also trained???"""
            train_loss = model_pretrain(model, model_optimizer, criterion, train_dl, config, device)
            print(f'\nPre-training Epoch : {epoch}', f'Train Loss : {train_loss:.4f}')
            # metrics_dict = eval_classification(model, train_data, train_labels, test_data, test_labels, config, classifier, device, fraction=None)
            # print(metrics_dict)

        # chkpoint = {'model_state_dict': model.state_dict()}
        torch.save(model.state_dict(), "test_run/{}/seed{}_pretrain_model.pt".format(config.dataset, seed))
    return train_loss


def model_finetune(model, model_optimizer, train_dl, config, device, classifier, classifier_optimizer):
    global labels, pred_numpy, fea_concat_flat
    model.train()
    classifier.train()

    total_loss = []
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_dl):
        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        # """if random initialization:"""
        # model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        # classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        # print('h_t shape:', h_t.shape)
        # print('h_t_aug shape:', h_t_aug.shape)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), \
            nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)  #

        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        # fea_concat = torch.cat((z_t, z_f), dim=1)
        fea_concat = torch.cat((z_t, z_t), dim=1)
        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss_p = criterion(predictions, labels)

        lam = 0.1
        loss = loss_p  # + l_TF + lam * (loss_t + loss_f)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

    ave_loss = torch.tensor(total_loss).mean()
    # print(' Finetune: loss = %.4f'% (ave_loss))

    return ave_loss


def finetune_training(model, model_optimizer, train_dl, valid_dl, config, device, classifier, classifier_optimizer,
                      fraction=None, seed=42):
    print('Fine-tune on Fine-tuning set')
    epoch_loss_list, epoch_f1_list = [], []
    global emb_finetune, label_finetune, emb_test, label_test

    for epoch in range(1, config.num_epoch_finetune + 1):

        train_loss = model_finetune(model, model_optimizer, train_dl, config, device, classifier, classifier_optimizer)
        epoch_loss_list.append(train_loss)
        print("Epoch number: {}, Loss: {}".format(epoch, epoch_loss_list[-1]))
        # save best fine-tuning model""
        f1, _, _, _, _, _, _, _ = model_test(model, valid_dl, config, device, classifier, ft_train=True)
        if len(epoch_f1_list) == 0 or f1 > max(epoch_f1_list):
            print('update fine-tuned model')
            if fraction:
                torch.save(model.state_dict(),
                           f"test_run/{config.dataset}/seed{seed}_max_f1_{fraction}_encoder_model.pt")
                torch.save(classifier.state_dict(),
                           f"test_run/{config.dataset}/seed{seed}_max_f1_{fraction}_encoder_classifier.pt")
            else:
                torch.save(model.state_dict(), f"test_run/{config.dataset}/seed{seed}_max_f1_encoder_model.pt")
                torch.save(classifier.state_dict(),
                           f"test_run/{config.dataset}/seed{seed}_max_f1_encoder_classifier.pt")
        epoch_f1_list.append(f1)

    return epoch_f1_list, epoch_loss_list


def model_test(model, test_dl, config, device, classifier, ft_train=False):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss()  # the loss for downstream classifier
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _, data_f, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions_test = classifier(fea_concat)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)

            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels, num_classes=config.num_classes)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,
                                       average="macro", multi_class="ovr")
            except:
                auc_bs = np.float(0)
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
            if math.isnan(prc_bs):
                prc_bs = np.float(0)
            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_acc.append(acc_bs)
            total_auc.append(auc_bs)
            total_prc.append(prc_bs)

            total_loss.append(loss.item())
            pred = predictions_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

            matrix = confusion_matrix(pred.detach().cpu().numpy(), labels.data.cpu().numpy())
            print(matrix)

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
    # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    if ft_train:
        print('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'
              % (acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100))
    emb_test_all = torch.cat(tuple(emb_test_all))
    return F1, total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance


def finetune_test(model, test_dl, config, device, classifier, fraction=None, seed=42):
    metrics = {}
    performance_list = []
    for epoch in range(1, config.num_epoch_finetune + 1):
        if fraction:
            model.load_state_dict(
                torch.load(f"test_run/{config.dataset}/seed{seed}_max_f1_{fraction}_encoder_model.pt"))
            classifier.load_state_dict(
                torch.load(f"test_run/{config.dataset}/seed{seed}_max_f1_{fraction}_encoder_classifier.pt"))
        else:
            model.load_state_dict(torch.load(f"test_run/{config.dataset}/seed{seed}_max_f1_encoder_model.pt"))
            classifier.load_state_dict(
                torch.load(f"test_run/{config.dataset}/seed{seed}_max_f1_encoder_classifier.pt"))

        _, _, _, _, _, _, _, performance = model_test(model, test_dl, config, device, classifier, ft_train=False)
        performance_list.append(performance)
    performance_array = np.array(performance_list)
    best_performance = performance_array[np.argmax(performance_array[:, 0], axis=0)]
    """print('Best Testing Performance: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f '
          '| AUPRC=%.4f' % (best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                            best_performance[4], best_performance[5]))"""

    metrics['Accuracy'] = best_performance[0]
    metrics['Precision'] = best_performance[1]
    metrics['Recall'] = best_performance[2]
    metrics['F1'] = best_performance[3]
    metrics['AUROC'] = best_performance[4]
    metrics['AUPRC'] = best_performance[5]

    return metrics
