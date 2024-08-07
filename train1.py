import torch
import collections
from model1 import build_ConvNet
import h5py
import argparse,os
import numpy as np
import pandas as pd
import logging
from typing import Literal
from datetime import datetime
from scipy.stats import pearsonr
def parser_args():
    parser = argparse.ArgumentParser(description="Argument parser for model training")
    parser.add_argument(
        '--data_cache_dir',
        type=str,
        required=True,
        help='The directory where the data is saved'
    )
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        required=True,
        help='The directory where the model will be saved'
    )
    parser.add_argument('--model_path', default=None, type=str,
                        help='The model checkpoint for initialization')
    parser.add_argument('--epochs', default=10, type=int,help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--log1p', default=True, action='store_false')
    parser.add_argument('--seqLen', default=2000, type=int, help='The length of the input sequence, which should be divisible by 100')

    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args

def concatenate_fa(fasta_file: str) -> str:
    '''
    concatenate the sequence in a fasta file into one string
    '''
    concat_dna=''
    with open(fasta_file,'r') as f:
        for line in f:
            if line.strip().startswith('>'):
                continue
            concat_dna+=line.strip()
    return concat_dna

def fatonumpy(fa_sequence: str) -> np.ndarray:
    '''
    :param fa_sequence: fasta sequence in 'ACGT'
    :return: one-hot encoded sequence in numpy array

    convert DNA sequence in string to one-hot encoded sequence
    '''
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot_seq = np.zeros((len(fa_sequence), len(nucleotide_to_index)))
    index_array = np.array([nucleotide_to_index.get(nuc, -1) for nuc in fa_sequence])
    nonN_indices = index_array >= 0
    one_hot_seq[np.arange(len(fa_sequence))[nonN_indices], index_array[nonN_indices]] = 1
    return one_hot_seq.T

def load_dataset(data_path: str):
    '''
    load the train/test/validation dataset
    '''

    pdfile = pd.read_csv(data_path)
    train_sets=np.array(pdfile[pdfile['split']=='train'].iloc[:,-3:])
    test_sets = np.array(pdfile[pdfile['split']=='test'].iloc[:,-3:])
    valid_sets = np.array(pdfile[pdfile['split']=='valid'].iloc[:,-3:])
    logging.info(f'Train/valid/test size: {len(train_sets)} , {len(valid_sets)},{len(test_sets)} \n')
    return train_sets,valid_sets,test_sets

def padSeq(seq: str, max_length: int, side: Literal['right', 'left'], pad_token:str ='N') -> str:
    '''
    pad the sequence with 'N' if the current sequence length is smaller than the input seqyence length
    '''

    if not len(seq)<max_length:
        raise ValueError('Sequence padding is not needed')
    if side not in {'right', 'left'}:
        raise ValueError("sequence padding is in the either 'right' or 'left'")
    pad_length=max_length-len(seq)
    pad_seq=pad_token*pad_length
    if side=='left':
        seq=pad_seq+seq
    else:
        seq=seq+pad_seq
    return seq

def shuffleData(data):
    indices=np.arange(len(data))
    np.random.shuffle(indices)
    return data[indices]

def main():
    args =get_args()

    if args.seqLen%100:
        raise ValueError('The input length is not divisible by 100.')


    data_cache_dir=os.path.abspath(args.data_cache_dir)
    if not os.path.exists(args.model_cache_dir):
        os.mkdir(args.model_cache_dir)
    model_cache_dir = os.path.abspath(args.model_cache_dir)

    if not os.path.isdir(os.path.join(data_cache_dir, 'logging')):
        os.mkdir(os.path.join(data_cache_dir, 'logging'))
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # log_file_name = f"train_{timestamp}.log"
    logging.basicConfig(filename=os.path.join(data_cache_dir, 'logging', f'train_{args.seqLen}_{args.log1p}.log'), level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}...')



    fasta_sequences={}
    dna_length={}

    for chrom in [1,2,3,4,5]:
        tags=''
        fasta_file=os.path.join(data_cache_dir,'Arabidopsis_thaliana.TAIR10.dna'+tags+'.chromosome.'+str(chrom)+'.fa')
        if not os.path.exists(fasta_file):
            logging.info(f'Download the fasta file of chromosome {chrom}... \n')
            download_link='https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-59/fasta/arabidopsis_thaliana/dna/'
            os.system('wget -O - '+download_link+
                      'Arabidopsis_thaliana.TAIR10.dna'+tags+'.chromosome.'+str(chrom)+'.fa.gz | gunzip -c > '+ fasta_file)
        fasta_sequences[chrom]=concatenate_fa(fasta_file)
        logging.info(f'Lenth of chromosome {chrom} is {len(fasta_sequences[chrom])}')
        dna_length[chrom]=len(fasta_sequences[chrom])

    expression_file=h5py.File(os.path.join(data_cache_dir,'arabidopsis_expression_data.h5'))
    targ_exression_matrix= np.array(expression_file['FPKM_matrix'])

    if args.log1p:
        targ_exression_matrix=np.log1p(targ_exression_matrix)

    gene_label_names= expression_file['chrom_geneid'][:,1].astype('str')
    gene_names_indices_dict={}
    for idx,n in enumerate(gene_label_names):
        gene_names_indices_dict[n]=idx

    model=build_ConvNet(n_features=targ_exression_matrix.shape[-1],model_path=args.model_path)
    model.to(device)

    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr= args.lr
                                ,weight_decay=1e-4)


    def createBatchinput(geneNames,Chroms,TSSloc,padLen=args.seqLen//2):
        '''
            Generate the inputs for each batch
            padLen equals half of the input sequence length
        '''
        batchInput=np.zeros((len(Chroms),4, padLen*2))
        for i in range(len(Chroms)):
            ifpad = None
            seqStart,seqEnd=TSSloc[i]-padLen, TSSloc[i]+padLen
            if seqStart<0:
                seqStart=0
                ifpad='left'
            elif seqEnd > dna_length[Chroms[i]]:
                seqEnd=dna_length[Chroms[i]]
                ifpad='right'
            seqinstr=fasta_sequences[Chroms[i]][seqStart:seqEnd]
            if ifpad is not None:
                seqinstr=padSeq(seqinstr, padLen*2, side=ifpad)
            batchInput[i,:,:]=fatonumpy(seqinstr)
        return torch.tensor(batchInput).float().to(device)

    def createBatchTarget(geneNames):
        '''
        Generate the expression targets for each batch
        '''
        targIndices=[]
        for gn in geneNames:
            sampleidx=gene_names_indices_dict.get(gn,None)
            if sampleidx is None:
                raise ValueError('Gene name error')
            targIndices.append(sampleidx)
        batchTarg=targ_exression_matrix[np.array(targIndices)]
        return torch.tensor(batchTarg).float().to(device)



    def prepare_inputs(dataChunk):
        '''
        Prepare model inputs and targets
        '''

        geneNames, Chroms,TSSloc= dataChunk[:,0],dataChunk[:,1].astype('int'),dataChunk[:,2].astype('int')
        batchInput=createBatchinput(geneNames,Chroms,TSSloc)
        batchTarget=createBatchTarget(geneNames)
        return batchInput,batchTarget

    def adjust_learning_rate(optimizer, epoch):
        if epoch < 10:
            lr = args.lr
        else:
            lr = 0.2*args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train_sets, valid_sets, test_sets=load_dataset(os.path.join(data_cache_dir,'arabidopsis_expression_data_split.csv'))

    bestScore=0
    for epoch in range(args.epochs):
        print(f'start training epoch {epoch}')
        model.train()
        training_loss = []
        train_sets=shuffleData(train_sets)
        adjust_learning_rate(optimizer, epoch)

        for chunki in range(0, len(train_sets),args.bs):

            data_chunk= train_sets[chunki: chunki+args.bs]
            bsInput,bsTarget=prepare_inputs(data_chunk)
            if epoch==0 and chunki==0:
                print(f'input size: {bsInput.shape}')
            bsOut=model(bsInput)
            loss=criterion(bsOut,bsTarget)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
        logging.info(f'Epoch: {epoch}, Training Loss: {np.mean(training_loss)} \n')

        logging.info('Start evaluation steps ... \n')
        model.eval()
        vpreds,vtargs=[],[]
        for chunki in range(0, len(valid_sets),args.bs):
            data_chunk= valid_sets[chunki: chunki+args.bs]
            bsInput,bsTarget=prepare_inputs(data_chunk)
            with torch.no_grad():
                bsOut=model(bsInput)
                loss=criterion(bsOut,bsTarget)
            vpreds.append(bsOut.cpu().data.detach().numpy())
            vtargs.append(bsTarget.cpu().data.detach().numpy())
        vpres=np.vstack(vpreds)
        vtargs=np.vstack(vtargs)
        validScore=pearsonr(vpres.flatten(),vtargs.flatten())[0]
        logging.info(f'Epoch: {epoch}, Valid Pearson: {validScore} \n')

        if validScore>bestScore:
            bestScore=validScore
            logging.info('Save model ... \n')
            torch.save(model.state_dict(),os.path.join(model_cache_dir,'ara.pt'))

        vpreds, vtargs = [], []
        for chunki in range(0, len(test_sets), args.bs):
            data_chunk = test_sets[chunki: chunki + args.bs]
            bsInput, bsTarget = prepare_inputs(data_chunk)
            with torch.no_grad():
                bsOut = model(bsInput)
                loss = criterion(bsOut, bsTarget)
            vpreds.append(bsOut.cpu().data.detach().numpy())
            vtargs.append(bsTarget.cpu().data.detach().numpy())
        vpres = np.vstack(vpreds)
        vtargs = np.vstack(vtargs)
        testScore = pearsonr(vpres.flatten(), vtargs.flatten())[0]
        logging.info(f'Epoch: {epoch}, Test Pearson: {testScore} \n')

        resultPath=os.path.join(data_cache_dir,'results')
        if not os.path.exists(resultPath):
            os.mkdir(resultPath)
        np.save(os.path.join(data_cache_dir,'results','ara_pred.npy') ,vpres)
        np.save(os.path.join(data_cache_dir, 'results', 'ara_targ.npy'), vtargs)



if __name__=='__main__':
    main()