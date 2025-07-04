package models

type ArticleRequest struct {
	Topic string `json:"topic"`
	Limit int    `json:"limit"`
}

type Article struct {
	Title  string `json:"title"`
	URL    string `json:"url"`
	Author string `json:"author"`
}

type ArticleResponse struct {
	Articles []Article `json:"articles"`
}
