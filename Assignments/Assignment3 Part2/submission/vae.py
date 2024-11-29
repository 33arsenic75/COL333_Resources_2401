import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import norm
import sys
import csv


torch.manual_seed(37)
np.random.seed(42)
NM_EPOCH = 100
one, zero = 1, 0


class ImageDataset(torch.utils.data.Dataset):
    # Custom dataset for selected digit classes#
    def __init__(self, data_path, target_classes=(1, 4, 8)):
        loaded = np.load(data_path)
        mask = np.isin(loaded["labels"], target_classes)
        self.data = torch.tensor(loaded["data"][mask] / 255.0, dtype=torch.float32)
        self.targets = torch.tensor(loaded["labels"][mask], dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderNetwork(nn.Module):
    # Encoder network matching the original architecture exactly#
    def __init__(self):
        super(EncoderNetwork, self).__init__()

        """ Layer to compute mean of the latent distribution """
        self.mean_layer = nn.Linear(400, 2)

        """ Layer to compute log variance of the latent distribution """
        self.logvar_layer = nn.Linear(400, 2)

        """ Define fully connected layers """
        self.input_layer = nn.Linear(784, 400)

    def forward(self, input_data):
        """Flatten input data to a single dimension"""
        flattened_data = input_data.view(-1, 784)
        """ Pass through hidden layer with ReLU activation """
        hidden_representation = torch.relu(self.input_layer(flattened_data))
        """ Compute mean and log variance for the latent distribution """
        mean_output = self.mean_layer(hidden_representation)
        log_variance_output = self.logvar_layer(hidden_representation)

        return mean_output, log_variance_output


class DecoderNetwork(nn.Module):
    # Decoder network to reconstruct images from latent space#
    def __init__(self):
        super().__init__()
        """Layer to map hidden layer to output layer"""
        self.fc4 = nn.Linear(400, 784)
        """Layer to map latent space to hidden layer"""
        self.fc3 = nn.Linear(2, 400)
        """Layer to map latent space to hidden layer"""

    def forward(self, z):
        """Pass through hidden layer with ReLU activation"""
        h3 = torch.relu(self.fc3(z))
        """ Pass through output layer with sigmoid activation """
        return torch.sigmoid(self.fc4(h3))


class ModernVAE(nn.Module):
    # VAE implementation to compress and reconstruct images #
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNetwork()
        self.decoder = DecoderNetwork()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std * one + zero + mu + zero * one
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Flatten x to match original architecture #
        x = x.view(-1, 784)
        KLD, BCE = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ), F.binary_cross_entropy(recon_x, x, reduction="sum")
        # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return BCE + KLD


def train_vae(model, train_loader, optimizer, device="cuda", epochs=10):
    model.train()
    for epoch_num in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = len(train_loader.dataset)
        """ Iterate over each batch of data in the training loader """
        for batch_num, (input_data, _) in enumerate(train_loader):
            input_data = input_data.to(device)
            """ Move the input data to the specified device (e.g., GPU) """
            optimizer.zero_grad()
            reconstructed, mean, log_variance = model(input_data)
            """ Pass the input data through the model to get the reconstructed output """
            batch_loss = model.loss_function(
                reconstructed, input_data, mean, log_variance
            )
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        avg_epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch_num}: Average Loss = {avg_epoch_loss:.4f}")


class GaussianMixtureModel:
    # Enhanced Gaussian Mixture Model implementation.#
    def __init__(self, n_components=3, tol=1e-3, max_iter=100):
        # Initialization with provided number of components and convergence criteria #
        self.n_components = n_components * one + zero
        self.tol = tol * one
        # Convergence threshold for early stopping#
        self.max_iter = max_iter
        #  Maximum number of iterations allowed #
        self.means = None
        self.covariances = None
        self.priors = None
        self.class_map = {}

    def initialize(self, dataset, initial_centroids, labels=[1, 4, 8]):
        # Set initial values for means, covariances, and mixture weights#
        # Initialize the mean, covariance, and prior for each component #

        # self.means = initial_centroids.copy()#
        self.means = np.copy(initial_centroids)
        # self.covariances = np.array([np.cov(dataset.T) for _ in range(self.n_components)])#
        self.covariances = np.stack(
            [np.cov(dataset.T) for _ in range(self.n_components)], axis=0
        )
        # self.priors = np.full(self.n_components, 1 / self.n_components)#
        self.priors = np.full(self.n_components, 1 / self.n_components)

    def compute_responsibilities(self, dataset, labels):
        # Calculate the responsibility matrix for each component #
        responsibilities = np.zeros((len(dataset), self.n_components))
        for i in range(self.n_components):
            # Calculate the responsibility matrix for each component #
            responsibilities[:, i] = zero + one * self.priors[
                i
            ] * multivariate_normal.pdf(dataset, self.means[i], self.covariances[i])

        # Normalize responsibilities to ensure each row sums to 1 #
        return responsibilities / responsibilities.sum(axis=1, keepdims=True)

    def update_parameters(self, dataset, responsibilities):
        # Calculate effective number of data points for each component #
        total_resp = responsibilities.sum(axis=0)
        for i in range(self.n_components):
            self.priors[i] = (one * total_resp[i]) / (len(dataset) + zero)
            self.means[i] = (
                one * (responsibilities[:, i] @ dataset) / total_resp[i]
            ) + zero
            deviation = dataset * one - self.means[i] + zero
            self.covariances[i] = (
                (responsibilities[:, i] * deviation.T) @ deviation / total_resp[i]
            ) * one + zero

    def train(self, dataset, initial_centroids, labels):
        # Initialize parameters and perform EM algorithm to fit the GMM to data #
        self.initialize(dataset, initial_centroids, labels=labels)
        for iteration in range(self.max_iter):
            old_means = self.means.copy()
            # Check for convergence by comparing means of current and previous iteration #
            responsibilities = self.compute_responsibilities(dataset, labels)
            # Check for convergence by comparing means of current and previous iteration #
            self.update_parameters(dataset, responsibilities)
            # Check for convergence by comparing means of current and previous iteration #
            if np.linalg.norm(self.means - old_means) < self.tol:
                break
        # Map each component to the closest label based on the initial centroids #
        for i, mean in enumerate(self.means):
            # Check for convergence by comparing means of current and previous iteration #
            distances = [
                np.linalg.norm(mean - centroid) for centroid in initial_centroids
            ]
            # Check for convergence by comparing means of current and previous iteration #
            self.class_map[i] = labels[np.argmin(distances)]

        print("Component to Class Mapping:", self.class_map)

    def predict(self, sample_data):
        # Compute the component probabilities and predict class labels.#
        """Calculate density for each component and sample"""
        densities = np.array(
            [
                multivariate_normal.pdf(sample_data, mean=mu, cov=cov)
                for mu, cov in zip(self.means, self.covariances)
            ]
        ).T

        """ Assign each sample to the component with the highest density """
        component_indices = np.argmax(densities, axis=1)

        """ Map the components to class labels """
        predicted_labels = np.array([self.class_map[idx] for idx in component_indices])

        return predicted_labels


def compute_vae_loss(recon_x, x, mu, logvar):
    # Compute VAE loss with KL divergence#
    kld, bce = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ), nn.functional.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
    )
    return bce + kld


def train_model(model, data_loader, optimizer, num_epochs=NM_EPOCH, device="cuda"):
    # Function to train the Variational Autoencoder (VAE) model.#
    model.train()
    for epoch_idx in range(1, num_epochs + 1):
        cumulative_loss = 0.0 * one + zero
        total_samples = one * len(data_loader.dataset) + zero
        for inputs, _ in data_loader:
            """input is sent to the device"""
            inputs = inputs.to(device)
            """zero the gradients to avoid accumulation"""
            optimizer.zero_grad()
            """model the input data to get the reconstructed data"""
            reconstructions, latent_mu, latent_logvar = model(inputs)
            """compute the loss for the model"""
            batch_loss = compute_vae_loss(
                reconstructions, inputs, latent_mu, latent_logvar
            )
            """compute the gradients"""
            batch_loss.backward()
            """update the weights"""
            cumulative_loss += (batch_loss.item()) * one + zero
            """optimize the model"""
            optimizer.step()

        avg_epoch_loss = (cumulative_loss * one + zero) / (total_samples * one + zero)
        print(
            f"Epoch {epoch_idx*one+zero}/{num_epochs*one+zero} - Average Loss: {avg_epoch_loss*one+zero:.4f}"
        )


def visualize_reconstructions(model, dataloader, num_samples=10, device="cuda"):
    """Visualize original vs reconstructed images"""
    model.eval()
    """ Set the model to evaluation mode """
    data, labels = next(iter(dataloader))
    """ Get a batch of data from the dataloader """
    data = data.to(device)
    """ Move the data to the specified device (e.g., GPU) """
    recon, _, _ = model(data)
    """ Pass the data through the model to get reconstructions """
    _, axes = plt.subplots(2, num_samples, figsize=(15, 4))
    for i in range(num_samples):
        """Iterate over the number of samples to plot"""
        axes[0, i].imshow(data[i].cpu().numpy().reshape(28, 28), cmap="gray")
        """ Plot the original image """
        axes[0, i].axis("off")
        """ Disable axis for better visualization """
        axes[1, i].imshow(recon[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
        """ Plot the reconstructed image """
        axes[1, i].axis("off")
        """ Disable axis for better visualization """
    plt.show()


def plot_latent_manifold(model, n=20, device="cuda"):
    # Plot 2D latent space manifold#
    figure = np.zeros((28 * n, 28 * n))
    # Create a grid of n x n points in the latent space #
    grid = norm.ppf(np.linspace(0.05, 0.95, n))
    # Plot 2D latent space manifold#
    model.eval()
    with torch.no_grad():
        """Disable gradient calculations for evaluation"""
        for i, yi in enumerate(grid):
            """Plot 2D latent space manifold"""
            for j, xi in enumerate(grid):
                """Plot 2D latent space manifold"""
                z = torch.tensor([[xi, yi]], device=device).float()
                """Plot 2D latent space manifold"""
                digit = model.decoder(z).cpu().view(28, 28).numpy()
                """Plot 2D latent space manifold"""
                figure[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = digit
    """Plot 2D latent space manifold"""
    plt.figure(figsize=(10, 10))
    """imshow() function displays data as an image"""
    plt.imshow(figure, cmap="gnuplot2")
    plt.axis("off")
    plt.show()


def visualize_latent_space(model, data_loader, device="cuda"):
    """Generate a 2D scatter plot of the latent space encoded by the model"""
    model.eval()
    latent_points = []
    class_labels = []

    """ Disable gradient calculations for evaluation """
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            latent_means, _ = model.encoder(inputs)

            """ Collect encoded representations and their corresponding labels """
            latent_points.append(latent_means.cpu().numpy())
            class_labels.extend(labels.numpy())

    """ Combine all batches into single arrays """
    latent_points = np.vstack(latent_points)
    class_labels = np.array(class_labels)

    """ Plot the encoded data in the latent space """
    plt.figure(figsize=(10, 8))
    """ Creates a new figure with a specified size for the plot (10x8 inches), """
    scatter_plot = plt.scatter(
        latent_points[:, zero],
        latent_points[:, one],
        c=class_labels,
        cmap="plasma",
        alpha=0.7 * one + zero,
        s=6 * one + zero,
    )
    # Creates a scatter plot for the latent space points.
    # - latent_points[:, 0] and latent_points[:, 1]: X and Y coordinates of each data point in the 2D latent space.
    # - alpha=0.7: Sets the transparency to 0.7, so overlapping points become more visible.
    # - s=6: Sets the size of each point to 6, which is optimal for visualization.
    plt.colorbar(scatter_plot, label="Digit Label")
    plt.title("2D Latent Space of Encoded Data")
    #     Adds a title to the plot that describes its purpose
    plt.xlabel("Latent Dimension 1")
    # Labels the X-axis to indicate it represents the first dimension of the latent space.
    plt.ylabel("Latent Dimension 2")
    # Labels the Y-axis to indicate it represents the second dimension of the latent space.
    # Show and save plot #
    plt.savefig("latent_space_distribution.png")
    plt.show()


def compute_class_centroids(model, data_loader, device="cuda"):
    """Calculate the centroid of each class in the latent space."""
    centroid_sums = {}
    label_counts = {}
    """ Initialize dictionaries to store sums and counts for each class """
    model.eval()
    """ Set the model to evaluation mode """
    with torch.no_grad():
        """Disable gradient calculations for evaluation"""
        for batch_data, batch_labels in data_loader:
            """Iterate over each batch of data and labels in the data loader"""
            batch_data = batch_data.to(device)
            """ Move the batch of data to the specified device (e.g., GPU) """
            latent_means, _ = model.encoder(batch_data)
            """ Pass the data through the encoder to get the latent mean """
            latent_means = latent_means.cpu().numpy()
            """ Move the latent means to the CPU and convert to a numpy array """
            for idx, lbl in enumerate(batch_labels):
                # Iterate over each data point and its label #
                lbl = lbl.item()
                if lbl in centroid_sums:
                    centroid_sums[lbl] += latent_means[idx]
                    label_counts[lbl] += 1
                else:
                    centroid_sums[lbl] = latent_means[idx]
                    label_counts[lbl] = 1

    centroids = np.array(
        [
            centroid_sums[class_label] / label_counts[class_label]
            for class_label in sorted(centroid_sums.keys())
        ]
    )
    unique_labels = sorted(centroid_sums.keys())
    return centroids, unique_labels


def visualize_gmm(gmm, data_points, data_labels):
    """Plot GMM components along with data distribution in the latent space."""
    plt.figure(figsize=(10, 8))
    """ Scatter plot for data points with distinct labels """
    one = 1
    zero = 0
    scatter_plot = plt.scatter(
        data_points[:, zero],
        data_points[:, one],
        c=data_labels,
        cmap="plasma",
        s=12 * one + zero,
        alpha=0.6 * one + zero,
        marker=".",
    )
    # Loop through each component in the GMM model to plot its covariance ellipse #
    for idx, (mean_vector, covariance_matrix) in enumerate(
        zip(gmm.means, gmm.covariances)
    ):
        # Loop through each component in the GMM model to plot its covariance ellipse #
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Eigenvalues and eigenvectors of the covariance matrix #
        rotation_angle = np.degrees(
            np.arctan2(*eigenvectors[:, 0 * one + zero][:: -1 * one + zero])
        )
        # Ellipse width and height derived from eigenvalues #
        ellipse_width, ellipse_height = (2 * one + zero) * np.sqrt(eigenvalues)
        # Create and plot the ellipse representing the GMM component #
        component_ellipse = Ellipse(
            xy=mean_vector,
            width=ellipse_width,
            height=ellipse_height,
            angle=rotation_angle,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
        plt.gca().add_patch(component_ellipse)

    plt.colorbar(scatter_plot, label="Digit Labels")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Data Distribution with GMM Component Ellipses")
    plt.savefig("gmm_component_distribution.png")
    plt.show()


def evaluate_classifier(labels_true, labels_pred):
    # Compute classification metrics#
    return {
        "accuracy": accuracy_score(labels_true, labels_pred),
        "precision": precision_score(labels_true, labels_pred, average="macro"),
        "recall": recall_score(labels_true, labels_pred, average="macro"),
        "f1": f1_score(labels_true, labels_pred, average="macro"),
    }


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_2d_manifold(model, latent_dim=2, n=20, digit_size=28, device="cuda"):
    """Plot the 2D manifold of the latent space"""
    figure = np.zeros((digit_size * n, digit_size * n))
    """ Create a grid of n x n points in the latent space """
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    """ Create a grid of n x n points in the latent space """
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    """ Create a grid of n x n points in the latent space """
    model.eval()
    """ Set the model to evaluation mode """
    with torch.no_grad():
        """Disable gradient calculations for evaluation"""
        for i, yi in enumerate(grid_x):
            """Iterate over the grid points in the x-direction"""
            for j, xi in enumerate(grid_y):
                """Iterate over the grid points in the y-direction"""
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                """ Ensure the decoder output is reshaped correctly (for example, as 28x28 image) """
                digit = (
                    model.decoder(z_sample).cpu().view(digit_size, digit_size).numpy()
                )
                """ Check if the decoder is generating the expected output """
                if digit.shape != (digit_size, digit_size):
                    print(f"Warning: Unexpected shape {digit.shape} from decoder")
                # Place the generated digit in the correct location in the grid #
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit
    """ Plot the generated digits in the latent space """
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gnuplot2")
    plt.axis("off")
    plt.show()
    """ Save the plot to a file """
    plt.savefig("Generation.png")


from skimage.metrics import structural_similarity as ssim


def show_reconstruction(model, val_loader):
    """Display original and reconstructed images from the validation set and save them"""
    model.eval()
    all_reconstructed_images = []
    all_original_images = []

    mse_values = []
    ssim_values = []

    for batch_idx, (data, labels) in enumerate(val_loader):
        data = data.to(device)
        recon_data, _, _ = model(data)

        # Collect original and reconstructed images for the entire dataset
        original_images = (
            data.cpu().numpy()
        )  # Shape: (batch_size, 1, 28, 28) or (batch_size, 28, 28)
        reconstructed_images = (
            recon_data.cpu().detach().numpy()
        )  # Shape: (batch_size, 784)

        all_original_images.append(original_images)
        all_reconstructed_images.append(reconstructed_images)

        # Calculate MSE and SSIM for the current batch
        for orig, recon in zip(original_images, reconstructed_images):
            orig_reshaped = orig.squeeze()  # Ensure (28, 28) shape
            recon_reshaped = recon.reshape(
                28, 28
            )  # Reshape flattened (784,) -> (28, 28)
            mse_values.append(np.mean((orig_reshaped - recon_reshaped) ** 2))
            ssim_values.append(
                ssim(
                    orig_reshaped,
                    recon_reshaped,
                    data_range=orig_reshaped.max() - orig_reshaped.min(),
                )
            )

    # Combine all batches into single arrays
    all_original_images = np.concatenate(
        all_original_images, axis=0
    )  # Shape: (total_images, 1, 28, 28) or (total_images, 28, 28)
    all_reconstructed_images = np.concatenate(
        all_reconstructed_images, axis=0
    )  # Shape: (total_images, 784)

    # Ensure original images have shape (total_images, 28, 28)
    if (
        all_original_images.ndim == 4 and all_original_images.shape[1] == 1
    ):  # Check for single channel
        all_original_images = all_original_images.squeeze(1)  # Remove channel dimension

    # Reshape reconstructed images to match original dimensions
    all_reconstructed_images = all_reconstructed_images.reshape(
        -1, 28, 28
    )  # Shape: (total_images, 28, 28)

    # Save reconstructed images as a NumPy array
    np.savez(
        "vae_reconstructed.npz",
        original_images=all_original_images,
        reconstructed_images=all_reconstructed_images,
    )

    # Calculate and display average metrics
    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Plot the first `n` images for visualization
    n = min(20, len(all_original_images))  # Display at most 10 samples
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        # Original image
        axes[0, i].imshow(all_original_images[i], cmap="gray")
        axes[0, i].axis("off")

        # Reconstructed image
        reconstructed = all_reconstructed_images[i]
        axes[1, i].imshow(reconstructed, cmap="gray")
        axes[1, i].axis("off")

    plt.show()
    plt.savefig("Reconstruction.png")


def extract_latent_vectors(model, dataloader, device="cuda"):
    """Extracts latent vectors from a trained model using data from the given dataloader."""
    model.eval()
    latents = []
    labels = []
    """ List to collect latent vectors for each batch """
    """ Set the model to evaluation mode """
    """ List to collect labels corresponding to each latent vector """
    with torch.no_grad():
        """Disable gradient calculation"""
        for data, label in dataloader:
            """Move data to the specified device (e.g., GPU)"""
            data = data.to(device)
            """ Pass data through the encoder to get the latent mean (mu) """
            mu, _ = model.encoder(data)
            """ Move mu to CPU and append it to the latents list """
            latents.append(mu.cpu())
            """ Extend the labels list with labels for this batch, moving them to numpy format """
            labels.extend(label.numpy())

    # Concatenate the list of latent vectors into a single numpy array #
    latents = torch.cat(latents).numpy()
    # Return the latent vectors and labels #
    return latents, labels


def plot_latent_space(gmm, latents, labels, n_clusters=3):
    # Plot the 2D latent space #
    plt.figure(figsize=(8, 6))
    """ Create a new figure with a specified size for the plot """
    plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap="viridis", s=2, alpha=0.7)
    # Scatter plot for the latent space points
    plt.colorbar()
    plt.title("VAE Latent Space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()

    # Fit a Gaussian Mixture Model to identify clusters #
    # gmm = GaussianMixtureModel() #
    # gmm.fit(latents) #
    cluster_labels = gmm.predict(latents)

    # Plot the clusters #
    plt.figure(figsize=(8, 6))
    plt.scatter(
        latents[:, 0], latents[:, 1], c=cluster_labels, cmap="tab10", s=2, alpha=0.7
    )
    plt.title("Latent Space Clusters")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()
    plt.savefig("scatter.png")

    return gmm


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernVAE().to(device)

    if len(sys.argv) == 4:
        print("Reconstruction mode")
        test_data = ImageDataset(sys.argv[1])
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=False
        )
        model.load_state_dict(torch.load(sys.argv[3]))
        show_reconstruction(model, test_loader)

    elif len(sys.argv) == 5:
        print("Classification mode")
        test_data = ImageDataset(sys.argv[1])
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=False
        )
        model.load_state_dict(torch.load(sys.argv[3]))
        with open(sys.argv[4], "rb") as f:
            gmm = pickle.load(f)

        pred_labels = []
        true_labels = []

        model.eval()
        """ Disable gradient calculations to save memory and increase computation speed """
        with torch.no_grad():
            """Iterate over each batch of images and labels in the test data"""
            for images, labels in test_loader:
                """Move the batch of images to the specified device (e.g., GPU if available)"""
                images = images.to(device)
                """ Pass images through the encoder part of the model to obtain mean (mu) and log variance """
                """ Only 'mu' is used here to represent the encoded latent variables """
                mu, _ = model.encoder(images)

                predictions = gmm.predict(mu.cpu().numpy())
                pred_labels.extend(predictions)
                true_labels.extend(labels.numpy())

        metrics = evaluate_classifier(true_labels, pred_labels)
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1']:.2f}")

        with open("vae.csv", mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            # Ensures commas are used as delimiters #
            writer.writerow(["Predicted_Label"])
            writer.writerows(zip(pred_labels))
        print("Predictions saved to vae.csv")

    else:
        print("Training mode")
        train_data = ImageDataset(sys.argv[1])
        val_data = ImageDataset(sys.argv[2])
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_model(model, train_loader, optimizer)

        plot_latent_manifold(model)
        visualize_latent_space(model, train_loader)

        centroids, class_labels = compute_class_centroids(model, val_loader)

        batch_data, _ = next(iter(train_loader))
        batch_data = batch_data.to(device)
        with torch.no_grad():
            encoded_batch, _ = model.encoder(batch_data)
            encoded_batch = encoded_batch.cpu().numpy()

        gmm = GaussianMixtureModel()
        # Train the GMM model on the encoded latent representations#
        gmm.train(encoded_batch, centroids, class_labels)
        # `encoded_batch` - The batch of data points encoded into the latent space
        # `centroids` - Initial positions for GMM components, derived from class centroids
        batch_labels = []

        # Iterate through the training data loader to gather true labels for visualization
        for _, labels in train_loader:
            # Move labels to CPU and convert to numpy format #
            batch_labels.extend(labels.numpy())

        # Visualize the GMM components and the distribution of encoded data in the latent space #
        visualize_gmm(gmm, encoded_batch, batch_labels[: len(encoded_batch)])
        # `gmm` - The trained GMM model containing means and covariances for each component #
        # This visualization helps in assessing how well the GMM clusters align with  #
        # the actual data distribution in the latent space #

        torch.save(model.state_dict(), sys.argv[4])
        with open(sys.argv[5], "wb") as f:
            pickle.dump(gmm, f)

        plot_2d_manifold(model, latent_dim=2, n=20)
        latents, labels = extract_latent_vectors(model, train_loader)
        plot_latent_space(gmm, latents, labels, n_clusters=3)
