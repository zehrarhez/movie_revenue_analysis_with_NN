import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time
import pickle
API_KEY = 'ENTER YOUR API'
BASE_URL = "https://api.themoviedb.org/3/"


def get_popular_movie_ids(num_movies=100):
    url = f"{BASE_URL}movie/popular"
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": 1
    }

    movie_ids = []

    # Sayfa sayısı belirtilen sayıya ulaşıncaya kadar popüler filmleri çekiyoruz
    while len(movie_ids) < num_movies:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200:
            movies = data["results"]
            # Her sayfadan filmlerin ID'sini alıyoruz
            for movie in movies:
                movie_ids.append(movie["id"])

            # Sonraki sayfaya geçmek için sayfayı bir artırıyoruz
            params["page"] += 1
        else:
            print(f"Error: {data['status_message']}")
            break

    return movie_ids[:num_movies]


def format_duration_in_minutes(duration):
    parts = duration.split()
    total_minutes = 0
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.replace('h', '')) * 60
        elif 'min' in part:
            total_minutes += int(part.replace('min', ''))
    return total_minutes


def format_duration_in_hours_and_minutes(duration):
    hours = duration // 60
    minutes = duration % 60
    return f"{hours}h {minutes}min"


def get_movie_details(movie_id):
    url = f"{BASE_URL}movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        if data["budget"] == 0 or data["revenue"] == 0:
            return None
        # Filmin türlerini ayırıp bir liste olarak alalım
        genres = [genre["name"] for genre in data["genres"]]

        movie_info = {
            "ID": data["id"],
            "Title": data["title"],
            "Total Time (minutes)": format_duration_in_hours_and_minutes(data["runtime"]),
            "Budget": data["budget"],
            "Revenue": data["revenue"],
            "Genres": ", ".join(genres),  # Türleri virgülle ayırıp tek bir metin olarak kaydedelim
        }
        return movie_info
    else:
        print(f"Error: {data['status_message']}")
        return None


start_time = time.time()

if __name__ == "__main__":
    num_movies_to_fetch = 1500
    max_attempts = 5
    total_attempts = 0
    num_movies_fetched = 0
    movie_data = []

    while num_movies_fetched < num_movies_to_fetch and total_attempts < max_attempts:
        movie_ids = get_popular_movie_ids(num_movies_to_fetch * 2)  # Fetch double the number of desired movies

        if movie_ids:
            for movie_id in movie_ids:
                movie_info = get_movie_details(movie_id)
                if movie_info:
                    movie_data.append(movie_info)
                    num_movies_fetched += 1

                if num_movies_fetched >= num_movies_to_fetch:
                    break

        total_attempts += 1

    df = pd.DataFrame(movie_data)
    df.to_csv("movie_data.csv", index=False)
    print("Movie data exported to movie_data.csv")


end_time = time.time()
print(df.isnull().sum())
df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

# Format_duration fonksiyonunu kullanarak "Total Time (minutes)" sütununu dönüştürme
df["Total Time (minutes)"] = df["Total Time (minutes)"].apply(format_duration_in_minutes)
def total_data_fetching_time():
    total_time = end_time - start_time  # toplam süre
    print(f"1000 data çekme işlemi {total_time} saniye sürdü.")

# "Budget" ve "Revenue" özellikleri için kutu grafiği
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.boxplot(df["Budget"])
plt.title("Budget Box Plot")

plt.subplot(2, 1, 2)
plt.boxplot(df["Revenue"])
plt.title("Revenue Box Plot")

plt.show()

def detect_outliers(data_column):
    Q1 = data_column.quantile(0.25)
    Q3 = data_column.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data_column[(data_column < lower_bound) | (data_column > upper_bound)]
    return outliers

# Aykırı değerleri belirlemek için "Budget" ve "Revenue" özelliklerini kullanın
outliers_budget = detect_outliers(df["Budget"])
outliers_revenue = detect_outliers(df["Revenue"])

print("Aykırı Değerler - Budget:")
print(outliers_budget)

print("Aykırı Değerler - Revenue:")
print(outliers_revenue)

# Örnek olarak, "Budget" ve "Revenue" özellikleri için üst sınırı 99.9. yüzdelik dilimdeki değerle değiştirme
upper_limit_budget = df["Budget"].quantile(0.999)
df["Budget"] = df["Budget"].apply(lambda x: upper_limit_budget if x > upper_limit_budget else x)
upper_limit_revenue = df["Revenue"].quantile(0.999)
df["Revenue"] = df["Revenue"].apply(lambda x: upper_limit_revenue if x > upper_limit_revenue else x)

df["Total Time"] = df["Total Time (minutes)"].apply(format_duration_in_hours_and_minutes)

# Ölçeklendirme
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["Budget", "Revenue", "Total Time (minutes)"]])
df[["Budget", "Revenue", "Total Time (minutes)"]] = scaled_data

# Tür adları için ikili sütunlar oluşturma
df_genres = df["Genres"].str.get_dummies(sep=", ")

# Eski "Genres" sütununu çıkaralım
df.drop("Genres", axis=1, inplace=True)

# Ölçeklendirmeyi atladığımız için Total Time (minutes) sütununu kaldırma
df.drop("Total Time (minutes)", axis=1, inplace=True)

# Yeni tür sütunlarını DataFrame'e ekleyelim
df = pd.concat([df, df_genres], axis=1)

pd.set_option("display.max_columns", None)

# Yeni CSV dosyasına verileri aktar
csv_filename = "movie_data_with_preprocessing.csv"
df.to_csv(csv_filename, index=False)
print(f"Updated table exported to {csv_filename}")
end_time = time.time()
print("Updated Table:")
print(df)

# Read the CSV file
df = pd.read_csv("movie_data_with_preprocessing.csv")

# Convert the "Total Time (minutes)" column to numeric values (total minutes)
df["Total Time (minutes)"] = df["Total Time"].apply(format_duration_in_minutes)

# Create bar plots for "Budget," "Revenue," and "Total Time (minutes)" columns

# Set the style for the plots
sns.set(style="whitegrid")

# Create a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot the Budget column
sns.barplot(data=df, x="Title", y="Budget", ax=axes[0])
axes[0].set_title("Budget of Movies")
axes[0].set_xlabel("Movies")
axes[0].set_ylabel("Normalized Value")
axes[0].tick_params(axis='x', rotation=90)

# Plot the Revenue column
sns.barplot(data=df, x="Title", y="Revenue", ax=axes[1])
axes[1].set_title("Revenue of Movies")
axes[1].set_xlabel("Movies")
axes[1].set_ylabel("Normalized Value")
axes[1].tick_params(axis='x', rotation=90)

# Plot the Total Time (minutes) column
sns.barplot(data=df, x="Title", y="Total Time (minutes)", ax=axes[2])
axes[2].set_title("Total Time of Movies")
axes[2].set_xlabel("Movies")
axes[2].set_ylabel("Total Time (minutes)")
axes[2].tick_params(axis='x', rotation=90)

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()

total_data_fetching_time()
def save_scaler_to_pickle(scaler, filename):
    with open(filename, 'wb') as file:
        pickle.dump(scaler, file)
    print("scalers exported to .pkl file.")

# Replace 'scaler.pkl' with the desired filename for saving the scaler
save_scaler_to_pickle(scaler, 'scaler2.pkl')
