% Multi-View Clustering using Spherical k-Means.
% See: S. Bickel, T. Scheffer: Multi-View Clustering, ICDM 04.
% Hierachical clustering used to determine the initial centers for k-Means
% 
% @param view1 View number one, a data frame with the same row names as view2. All columns numeric. Entries are natural numbers, starting from 1.
% @param view2 View number two, a data frame with the same row names as view1. All columns numeric. Entries are natural numbers, starting from 1.
% @param k The maximum number of clusters to create
% @param startView The view on which to perform the initial E step, one of "view1", "view2"
% @param nthresh The number of iterations to run without improvement of the objective function
% @param doOutput Whether output to the console should be done
% @param doDebug Whether debug output to the console should be done (implies normal output)
% @param plotFile File name where the hierarchical clustering plot should be stored
% @return A list reporting the final clustering, with names finalIndices, agreementRate, indicesSv, indicesOv. They designate final cluster indices as a vector, as well as agreement rate of the two views, and the individual indices given by the two views over the course of iterations as data frames.

function [cluster_indicator] = mvc(view1, view2, K, nround)

    if nargin < 4
        nround = 20;
    end
    % First thing: convert rows to unit length
    view1 = view1./ repmat(sqrt(sum(view1.^2,2)),1, size(view1,2));
    view1(isnan(view1)) = 0;
    view2 = view2./ repmat(sqrt(sum(view2.^2,2)),1, size(view2,2));
    view2(isnan(view2)) = 0;
    startView = view1;
    otherView = view2;
    
    %Get k centers from startView
    [CIdx, initialCenters] = litekmeans(startView, K, 'Replicates', 10);

    %Main loop until objective function yields termination criterion
    % Uses lists to exchange views
    % Starts with processing otherView
    views{1} = otherView;
    views{2} = startView;
    centers{1} = [];
    centers{2} = initialCenters;
    indeces{1} = [];
    indeces{2} = CIdx;
    maxima{1} = -inf;
    maxima{2} = -inf;
    rounds{1} = 0;
    rounds{2} = 0;
    itCount = 0;
    t = 0;
    agreementRateDf = [];
    indicesSVDf = [];
    indicesOVDf = [];

    while true
        t = t+1;
        itCount = itCount +1;
        % M-Step: cluster centers
        centers{1} = calculateCenters(views{1}, indeces{2});
        % E-Step: cluster assignment for data
        indeces{1} = calculateIndeces(views{1}, centers{1});

        if (itCount == 2)
            itCount = 0;
            rounds{1} = rounds{1} +1;
            rounds{2} = rounds{2} +1;
            of2 = ofSkm(views{2}, centers{2}, indeces{2});
            of1 = ofSkm(views{1}, centers{1}, indeces{1});
            if of2 > maxima{2}
                maxima{2} = of2;
                rounds{2} = 0;
            end
            if of1 > maxima{1}
                maxima{1} = of1;
                rounds{1} = 0;
            end
        end
        if rounds{1} > nround && rounds{2} > nround
            break;
        end
        % switch
        views = swap(views);
        centers = swap(centers);
        indeces = swap(indeces);
        maxima = swap(maxima);
        rounds = swap(rounds);
    end
    consensusMeans = consensusMeansPerClVSkm(views{1}, views{2}, indeces{1}, indeces{2});
    cluster_indicator = assignFinIdxPerClSkm(views{1}, views{2}, consensusMeans);
end
function result = swap(x)
    result{1} = x{2};
    result{2} = x{1};
end
function centers = calculateCenters(view, index)
    centers = zeros(numel(unique(index)), size(view,2));
    for i = 1:numel(unique(index))
        centers(i,:) = mean(view(index==i,:));
        centers(i,:) = centers(i,:)./sqrt(sum(centers(i,:).^2));
    end
end
function indeces = calculateIndeces(view, centers)
    d = pdist2(view,centers);
    [~,indeces] = min(d,[],2);
end

function obj = ofSkm(view, centers, indeces)
    obj = 0;
    for i = 1:numel(unique(indeces))
        obj = obj + sum(view(indeces==i,:)*centers(i,:)');
    end
end
function concensusMeans = consensusMeansPerClVSkm(view1, view2, index1, index2)
    labels = intersect(unique(index1), unique(index2));
    for i = 1:numel(labels)
        sharedInJ = index1 == labels(i) & index2 == labels(i);
        mPerClV1 = mean(view1(sharedInJ,:));
        mPerClV1 = mPerClV1 ./ repmat(sqrt(sum(mPerClV1.^2,2)), 1, size(mPerClV1,2));
        mPerClV2 = mean(view2(sharedInJ,:));
        mPerClV2 = mPerClV2 ./ repmat(sqrt(sum(mPerClV2.^2,2)), 1, size(mPerClV2,2));
        concensusMeans{1}(i,:)= mPerClV1;
        concensusMeans{2}(i,:) = mPerClV2;
    end
end
function finalCIdx = assignFinIdxPerClSkm(view1, view2, mperClV)
    view1 = view1./repmat(sqrt(sum(view1.^2,2)), 1, size(view1,2));
    view1(isnan(view1)) = 0;
    view2 = view2./repmat(sqrt(sum(view2.^2,2)), 1, size(view2,2));
    view2(isnan(view2)) = 0;
    mPerClV1 = mperClV{1};
    mPerClV2 = mperClV{2};
    K = size(mPerClV1,1);
    n = size(view1,1);
    finalCIdx = zeros(1,n);
    sphericMin = inf*ones(n,1);
    for i = 1:K
        sphericVal = acos(view1 * mPerClV1(i,:)') + acos(view2 * mPerClV2(i,:)');
        minPat = sphericVal < sphericMin;
        sphericMin(minPat) = sphericVal(minPat);
        finalCIdx(minPat) = i;
    end
end
