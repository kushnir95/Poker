#pragma once
#include <cassert>
#include <vector>
#include <algorithm>

class Card
{

private:
	enum Suits{SPADE = 0, CLUB = 1, HEART = 2, DIAMOND = 3};
	enum Ranks{TWO = 2, THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8, NINE = 9, TEN = 10, JACK = 11, QUEEN = 12, KING = 13, ACE = 14};

	int suit;
	int rank;

public:

	Card();
	Card(int suit, int rank);
	
	int getSuit() const;
	int getRank() const;

	void setSuit(int newSuit);
	void setRank(int newRank);

	bool operator<(const Card& compCard);
	std::string getCombination(std::vector<Card> cardSet);
	std::string toString();

	~Card();
};

